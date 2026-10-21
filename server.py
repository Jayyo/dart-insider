"""
DART Insider web server.

Serves index.html and a small JSON API that turns OpenDART insider ownership
filings (임원ㆍ주요주주특정증권등소유상황보고서) into buy/sell records.
"""

import asyncio
import logging
import os
import random
import re
import time
import weakref
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import filings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# httpx logs every request URL at INFO level, and OpenDART URLs carry the API key.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("dart_insider")

ROOT = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9))

DART_API_KEY = os.getenv("DART_API_KEY", "").strip()
DART_BASE_URL = os.getenv("DART_BASE_URL", "https://opendart.fss.or.kr/api").strip().rstrip("/")

# OpenDART blocks clients that send bursts of requests, so each instance
# spaces its calls out and stops before the per-key daily quota (20,000).
DART_REQUESTS_PER_SECOND = 8
DART_DAILY_BUDGET = 15000
DOC_CONCURRENCY = 4
DAY_TIME_BUDGET = 12.0  # seconds per request; the client asks again for the rest
MAX_LOOKBACK_DAYS = 366
LOADS_PER_CLIENT = 90  # uncached days one client may trigger per 10 minutes

DART_ERRORS = {
    "010": (503, "등록되지 않은 OpenDART API 키입니다."),
    "011": (503, "사용할 수 없는 OpenDART API 키입니다."),
    "012": (502, "OpenDART가 이 서버의 IP를 허용하지 않습니다. 국내 리전이나 DART_BASE_URL 프록시가 필요합니다."),
    "020": (429, "OpenDART 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."),
    "800": (503, "OpenDART 시스템 점검 중입니다."),
    "901": (503, "OpenDART API 키의 이용 기간이 만료되었습니다."),
}

app = FastAPI(title="DART Insider", docs_url=None, redoc_url=None, openapi_url=None)


class UpstreamError(Exception):
    def __init__(self, message, status=502, fatal=True):
        super().__init__(message)
        self.message = message
        self.status = status
        self.fatal = fatal


class TTLCache:
    def __init__(self, max_items):
        self.items = OrderedDict()
        self.max_items = max_items

    def get(self, key):
        hit = self.items.get(key)
        if hit is None:
            return None
        expires, value = hit
        if expires is not None and expires < time.monotonic():
            del self.items[key]
            return None
        self.items.move_to_end(key)
        return value

    def put(self, key, value, ttl=None):
        self.items[key] = (time.monotonic() + ttl if ttl else None, value)
        self.items.move_to_end(key)
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)


class Throttle:
    """Hands out evenly spaced time slots. No asyncio primitives, so it is
    safe to share across event loops."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.next_at = 0.0

    async def wait(self):
        now = time.monotonic()
        slot = max(now, self.next_at)
        self.next_at = slot + self.interval
        if slot > now:
            await asyncio.sleep(slot - now)


class ClientLimiter:
    """Caps how many uncached days one client can make this instance fetch."""

    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.hits = {}

    def allow(self, key):
        now = time.monotonic()
        recent = [t for t in self.hits.get(key, ()) if now - t < self.window]
        allowed = len(recent) < self.limit
        if allowed:
            recent.append(now)
        self.hits[key] = recent
        if len(self.hits) > 5000:
            self.hits = {k: v for k, v in self.hits.items() if v and now - v[-1] < self.window}
        return allowed


class DailyBudget:
    def __init__(self, limit):
        self.limit = limit
        self.day = None
        self.used = 0

    def spend(self):
        today = datetime.now(KST).date()
        if today != self.day:
            self.day, self.used = today, 0
        if self.used >= self.limit:
            raise UpstreamError("오늘 사용할 수 있는 OpenDART 호출량을 모두 사용했습니다.", 429)
        self.used += 1


dart_throttle = Throttle(DART_REQUESTS_PER_SECOND)
dart_budget = DailyBudget(DART_DAILY_BUDGET)
client_limiter = ClientLimiter(LOADS_PER_CLIENT, 600)

documents = TTLCache(3000)  # rcept_no -> parsed filing; filings never change
listings = TTLCache(400)  # day -> list.json rows
days = TTLCache(400)  # day -> finished response body

_inflight = weakref.WeakKeyDictionary()  # event loop -> {day: task}


def inflight_tasks():
    # Vercel may run each request in a fresh event loop, so tasks are tracked per loop.
    return _inflight.setdefault(asyncio.get_running_loop(), {})


def today_kst():
    return datetime.now(KST).date()


def day_ttl(day):
    age = (today_kst() - day).days
    if age <= 0:
        return 120  # filings for today are still coming in
    if age <= 7:
        return 1800  # amendments usually land within a few days
    return 6 * 3600


def client_key(request):
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def new_client():
    return httpx.AsyncClient(
        timeout=httpx.Timeout(8.0, connect=4.0),
        headers={"User-Agent": "dart-insider (+https://github.com/Jayyo/dart-insider)"},
        follow_redirects=False,
    )


async def dart_get(client, path, params):
    if not DART_API_KEY:
        raise UpstreamError("서버에 DART_API_KEY가 설정되어 있지 않습니다.", 503)
    for attempt in range(3):
        await dart_throttle.wait()
        dart_budget.spend()
        try:
            resp = await client.get(f"{DART_BASE_URL}/{path}", params={"crtfc_key": DART_API_KEY, **params})
        except httpx.TransportError as exc:
            log.warning("OpenDART %s failed (%s), attempt %d", path, type(exc).__name__, attempt + 1)
            if attempt == 2:
                raise UpstreamError("OpenDART에 연결할 수 없습니다.", 502) from exc
        else:
            if resp.status_code < 500:
                if resp.status_code != 200:
                    raise UpstreamError(f"OpenDART 응답 오류 (HTTP {resp.status_code})", 502)
                return resp
            log.warning("OpenDART %s returned HTTP %d, attempt %d", path, resp.status_code, attempt + 1)
            if attempt == 2:
                raise UpstreamError(f"OpenDART 응답 오류 (HTTP {resp.status_code})", 502)
        await asyncio.sleep(0.6 * (attempt + 1))


def dart_error(status, message):
    code, text = DART_ERRORS.get(status, (502, f"OpenDART 오류 {status}: {message or ''}".strip()))
    return UpstreamError(text, code)


async def list_filings(client, day):
    cached = listings.get(day)
    if cached is not None:
        return cached
    rows, page = [], 1
    ymd = day.strftime("%Y%m%d")
    while True:
        resp = await dart_get(client, "list.json", {
            "bgn_de": ymd,
            "end_de": ymd,
            "pblntf_detail_ty": "D002",  # 임원ㆍ주요주주특정증권등소유상황보고서
            "last_reprt_at": "Y",        # only the latest version of amended filings
            "page_count": 100,
            "page_no": page,
        })
        try:
            data = resp.json()
        except ValueError as exc:
            raise UpstreamError("OpenDART 목록 응답을 해석할 수 없습니다.", 502) from exc
        status = data.get("status")
        if status == "013":  # no filings
            break
        if status != "000":
            raise dart_error(status, data.get("message"))
        rows.extend(data.get("list") or [])
        if page >= int(data.get("total_page") or 1) or page >= 20:
            break
        page += 1
    items = [filings.list_item(r) for r in rows]
    listings.put(day, items, ttl=day_ttl(day))
    return items


async def fetch_document(client, rcept_no):
    resp = await dart_get(client, "document.xml", {"rcept_no": rcept_no})
    content = resp.content
    if not content.startswith(b"PK"):
        status = re.search(rb'"?status"?\s*[:>]\s*"?(\d{3})', content[:500])
        if status and status.group(1).decode() in DART_ERRORS:
            raise dart_error(status.group(1).decode(), "")
        raise UpstreamError(f"{rcept_no} 원문을 받을 수 없습니다.", 502, fatal=False)
    try:
        return filings.parse_document(filings.read_document_zip(content))
    except Exception as exc:
        log.warning("Could not parse %s: %s", rcept_no, exc)
        raise UpstreamError(f"{rcept_no} 원문을 해석할 수 없습니다.", 502, fatal=False) from exc


async def fetch_documents(client, items, deadline):
    """Fetch and parse documents until the deadline. Returns (parsed, failed)."""
    semaphore = asyncio.Semaphore(DOC_CONCURRENCY)

    async def one(item):
        async with semaphore:
            return await fetch_document(client, item["rcept_no"])

    tasks = {asyncio.ensure_future(one(item)): item for item in items}
    parsed, failed, waiting = {}, [], set(tasks)
    try:
        while waiting:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, waiting = await asyncio.wait(waiting, timeout=remaining, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                item, exc = tasks[task], task.exception()
                if exc is None:
                    parsed[item["rcept_no"]] = task.result()
                elif isinstance(exc, UpstreamError) and exc.fatal:
                    raise exc
                else:
                    failed.append(item)
    finally:
        for task in waiting:
            task.cancel()
        if waiting:
            await asyncio.wait(waiting)
        for task in tasks:
            if not task.cancelled():
                task.exception()  # mark as retrieved; the first fatal one was already raised
    return parsed, failed


async def load_day(day):
    deadline = time.monotonic() + DAY_TIME_BUDGET
    async with new_client() as client:
        items = await list_filings(client, day)
        parsed = {}
        todo = []
        for item in items:
            doc = documents.get(item["rcept_no"])
            if doc is None:
                todo.append(item)
            else:
                parsed[item["rcept_no"]] = doc
        # A retry may land on another instance; a different order lets it make progress there too.
        random.shuffle(todo)
        fetched, failed = await fetch_documents(client, todo, deadline)

    for rcept_no, doc in fetched.items():
        documents.put(rcept_no, doc)
    parsed.update(fetched)
    failed_ids = {item["rcept_no"] for item in failed}

    records, pending, other = [], [], 0
    for item in items:
        doc = parsed.get(item["rcept_no"])
        if doc is None:
            if item["rcept_no"] not in failed_ids:
                pending.append(item["rcept_no"])
        elif doc["buy"]["shares"] or doc["sell"]["shares"]:
            records.append(filings.public_record(item, doc))
        else:
            other += 1

    return {
        "date": day.isoformat(),
        "complete": not pending and not failed,
        "filings": records,
        "pending": pending,
        "failed": [{k: item[k] for k in ("rcept_no", "corp_name", "filer")} for item in failed],
        "summary": {"total": len(items), "trades": len(records), "other": other,
                    "failed": len(failed), "pending": len(pending)},
        "fetched_at": datetime.now(KST).isoformat(timespec="seconds"),
    }


def parse_day(value):
    if not re.fullmatch(r"\d{8}", value):
        raise HTTPException(400, "날짜는 YYYYMMDD 형식이어야 합니다.")
    try:
        day = datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise HTTPException(400, "존재하지 않는 날짜입니다.")
    today = today_kst()
    if day > today or day < today - timedelta(days=MAX_LOOKBACK_DAYS):
        raise HTTPException(400, f"최근 {MAX_LOOKBACK_DAYS}일 이내의 날짜만 조회할 수 있습니다.")
    return day


@app.middleware("http")
async def security_headers(request: Request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
    if request.url.path.startswith("/api/"):
        log.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code,
                 (time.monotonic() - started) * 1000)
    return response


@app.exception_handler(UpstreamError)
async def upstream_error(_request, exc):
    return JSONResponse({"detail": exc.message}, status_code=exc.status, headers={"Cache-Control": "no-store"})


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index():
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "public, max-age=0, s-maxage=600"})


@app.get("/api/filings/{value}")
async def get_filings(value: str, request: Request):
    day = parse_day(value)
    body = days.get(day)
    if body is None:
        tasks = inflight_tasks()
        task = tasks.get(day)
        if task is None:
            if not client_limiter.allow(client_key(request)):
                raise HTTPException(429, "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.")
            task = asyncio.ensure_future(load_day(day))
            tasks[day] = task
            task.add_done_callback(lambda _t, d=day: tasks.pop(d, None))
        body = await asyncio.shield(task)
        if body["complete"]:
            days.put(day, body, ttl=day_ttl(day))

    if not body["complete"]:
        cache = "no-store"  # the client asks again for pending or failed filings
    else:
        ttl = day_ttl(day)
        cache = f"public, max-age={min(ttl, 600) // 2}, s-maxage={max(ttl, 300)}, stale-while-revalidate={ttl * 4}"
    return JSONResponse(body, headers={"Cache-Control": cache})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8080")))
