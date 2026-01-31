"""
DART Insider Web Server
FastAPI backend with Supabase DB + DART API proxy
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta, date
import os
import logging
import time
import json
import zipfile
import io
import re

# Load .env file if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DART Insider API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    real_ip = forwarded_for.split(",")[0].strip() if forwarded_for else client_ip
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    logger.info(
        f"IP: {real_ip} | {method} {path} | Query: {query} | "
        f"Status: {response.status_code} | Time: {duration}ms"
    )
    return response

# Configuration
API_KEY = os.getenv("DART_API_KEY", "9361d74facc8c239f634b08c0f436192de5c14de")
BASE_URL = os.getenv("DART_BASE_URL", "http://124.56.74.13:5959/api")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zlhybabcuyigewbevqcp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
CRON_SECRET = os.getenv("CRON_SECRET", "dart-insider-cron-2024")

# Lazy-loaded Supabase client
_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        if not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_SERVICE_KEY not set")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


# RPT_RSN code mapping table
RPT_RSN_MAP = {
    "01": "장내매수(+)", "02": "장외매수(+)",
    "11": "장내매도(-)", "12": "장외매도(-)",
    "21": "유상신주취득(+)", "22": "신주인수권행사(+)",
    "23": "무상신주취득(+)", "24": "제3자배정유상증자(+)",
    "25": "스톡옵션행사(+)", "26": "전환권행사(+)",
    "27": "교환권행사(+)", "29": "신규상장(+)",
    "31": "신규선임(+)", "98": "기타(-)", "99": "기타(+)",
}


@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")


@app.get("/api/purchases")
async def get_purchases(
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)"),
    force_refresh: bool = Query(False, description="Force refresh from DART API")
):
    """
    Get executive stock purchases for a date range.
    1. Check Supabase DB first
    2. If no data found (or force_refresh), fetch from DART API and store in DB
    """
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y%m%d").date()
        end_dt = datetime.strptime(end_date, "%Y%m%d").date()

        if not force_refresh:
            # Try DB first
            db_result = await get_from_db(start_dt, end_dt)
            if db_result is not None:
                return db_result

        # DB miss or force_refresh: fetch from DART API
        logger.info(f"Fetching from DART API for {start_date}-{end_date}...")
        result = await fetch_from_dart(start_date, end_date)

        # Store in DB (background, non-blocking for response)
        if result["success"] and result["data"]:
            asyncio.create_task(store_to_db(result["data"], start_dt, end_dt))

        return result

    except Exception as e:
        logger.error(f"Error in get_purchases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_from_db(start_dt: date, end_dt: date) -> Optional[dict]:
    """Query Supabase DB for cached data within date range"""
    try:
        sb = get_supabase()

        # Check if all dates in range have been updated
        result = sb.table("update_log") \
            .select("update_date, records_count, created_at") \
            .gte("update_date", start_dt.isoformat()) \
            .lte("update_date", end_dt.isoformat()) \
            .eq("status", "success") \
            .execute()

        updated_dates = {row["update_date"] for row in result.data}
        # Get the most recent update timestamp
        last_updated_at = max((row["created_at"] for row in result.data), default=None) if result.data else None

        # Check coverage: count business days roughly
        # If we have at least some data, return it
        if not updated_dates:
            return None

        # Fetch purchases from DB
        purchases = sb.table("executive_purchases") \
            .select("*") \
            .gte("report_date", start_dt.isoformat()) \
            .lte("report_date", end_dt.isoformat()) \
            .order("shares", desc=True) \
            .execute()

        if not purchases.data:
            return None

        # Format response
        data = []
        for p in purchases.data:
            data.append({
                "corp_name": p["corp_name"],
                "stock_code": p.get("stock_code", ""),
                "market": p["market"],
                "exec_name": p["exec_name"],
                "position": p.get("position", ""),
                "shares": p["shares"],
                "rate": p.get("rate", ""),
                "report_date": p.get("report_date", ""),
                "total_shares": p.get("total_shares", 0),
                "reason": p.get("reason", ""),
                "rcept_no": p.get("rcept_no", ""),
            })

        stats = compute_stats(data)

        logger.info(f"DB HIT: {len(data)} records for {start_dt}-{end_dt}")
        return {
            "success": True,
            "data": data,
            "stats": stats,
            "source": "database",
            "last_updated": last_updated_at,
            "message": f"총 {len(data)}건의 임원 자사주 매입 데이터를 조회했습니다. (DB)"
        }

    except Exception as e:
        logger.error(f"DB query error: {e}")
        return None


async def store_to_db(data: list, start_dt: date, end_dt: date):
    """Store fetched data to Supabase DB"""
    try:
        sb = get_supabase()

        # Group by report_date
        by_date = {}
        for item in data:
            rd = item.get("report_date", "")
            if rd:
                by_date.setdefault(rd, []).append(item)

        for report_date, items in by_date.items():
            rows = []
            for item in items:
                # Parse report_date to proper format
                rd = item.get("report_date", "")
                if len(rd) == 8:
                    rd = f"{rd[:4]}-{rd[4:6]}-{rd[6:8]}"

                rows.append({
                    "corp_name": item["corp_name"],
                    "stock_code": item.get("stock_code", ""),
                    "market": item["market"],
                    "exec_name": item["exec_name"],
                    "position": item.get("position", ""),
                    "shares": item["shares"],
                    "rate": item.get("rate", ""),
                    "report_date": rd,
                    "total_shares": item.get("total_shares", 0),
                    "reason": item.get("reason", ""),
                    "rcept_no": item.get("rcept_no", ""),
                })

            if rows:
                # Upsert to avoid duplicates
                sb.table("executive_purchases").upsert(
                    rows,
                    on_conflict="corp_name,exec_name,report_date,shares"
                ).execute()

                # Parse date for update_log
                rd_parsed = report_date
                if len(rd_parsed) == 8:
                    rd_parsed = f"{rd_parsed[:4]}-{rd_parsed[4:6]}-{rd_parsed[6:8]}"

                sb.table("update_log").upsert(
                    {
                        "update_date": rd_parsed,
                        "records_count": len(rows),
                        "status": "success"
                    },
                    on_conflict="update_date"
                ).execute()

        logger.info(f"Stored {len(data)} records to DB for {start_dt}-{end_dt}")

    except Exception as e:
        logger.error(f"DB store error: {e}")


def compute_stats(data: list) -> dict:
    """Compute summary stats from purchase data"""
    companies = len(set(p["corp_name"] for p in data))
    executives = len(data)
    total_shares = sum(p["shares"] for p in data)

    company_shares = {}
    for p in data:
        company_shares[p["corp_name"]] = company_shares.get(p["corp_name"], 0) + p["shares"]

    top_company = ""
    top_company_shares = 0
    if company_shares:
        top_company = max(company_shares, key=company_shares.get)
        top_company_shares = company_shares[top_company]

    market_dist = {"코스피": 0, "코스닥": 0, "기타": 0}
    for p in data:
        m = p.get("market", "기타")
        if m in market_dist:
            market_dist[m] += p["shares"]
        else:
            market_dist["기타"] += p["shares"]

    return {
        "companies": companies,
        "executives": executives,
        "total_shares": total_shares,
        "top_company": top_company,
        "top_company_shares": top_company_shares,
        "market_distribution": market_dist,
    }


async def fetch_from_dart(start_date: str, end_date: str) -> dict:
    """Fetch executive purchase data from DART API (original logic)"""
    start_dt = start_date
    end_dt = end_date

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Get all disclosures
        disclosures = await get_all_disclosures(client, start_date, end_date)

        # Step 2: Filter for executive reports
        keywords = ["임원", "주요주주", "소유상황"]
        unique_corps = {}

        for d in disclosures:
            report_nm = d.get("report_nm", "")
            if any(kw in report_nm for kw in keywords):
                corp_code = d.get("corp_code")
                if corp_code and corp_code not in unique_corps:
                    unique_corps[corp_code] = {
                        "corp_name": d.get("corp_name"),
                        "corp_cls": d.get("corp_cls"),
                        "stock_code": d.get("stock_code", "")
                    }

        logger.info(f"Found {len(unique_corps)} companies with executive reports")

        # Step 3: Get executive stock ownership for each company
        all_purchases = []

        for i, (corp_code, corp_info) in enumerate(unique_corps.items()):
            try:
                stock_list = await get_executive_stock(client, corp_code)

                for item in stock_list:
                    rcept_dt = item.get("rcept_dt", "").replace("-", "")
                    if not rcept_dt or rcept_dt < start_dt or rcept_dt > end_dt:
                        continue

                    is_exec = item.get("isu_exctv_rgist_at", "") in ["등록", "등기임원", "비등기임원"]
                    ofcps = item.get("isu_exctv_ofcps", "")
                    has_position = ofcps and ofcps != "-" and len(ofcps) > 0

                    irds_cnt_str = item.get("sp_stock_lmp_irds_cnt", "0")
                    try:
                        irds_cnt = int(str(irds_cnt_str).replace(",", "").replace("-", "0"))
                    except:
                        irds_cnt = 0

                    if (is_exec or has_position) and irds_cnt > 0:
                        market = corp_info["corp_cls"]
                        if market == "Y":
                            market = "코스피"
                        elif market == "K":
                            market = "코스닥"
                        else:
                            market = "기타"

                        all_purchases.append({
                            "corp_code": corp_code,
                            "rcept_no": item.get("rcept_no", ""),
                            "corp_name": corp_info["corp_name"],
                            "stock_code": corp_info["stock_code"],
                            "market": market,
                            "exec_name": item.get("repror", ""),
                            "position": ofcps,
                            "shares": irds_cnt,
                            "rate": item.get("sp_stock_lmp_rate", ""),
                            "report_date": item.get("rcept_dt", ""),
                            "total_shares": 0
                        })

                if (i + 1) % 50 == 0:
                    await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"Error fetching {corp_code}: {e}")
                continue

        # Fetch total shares
        unique_corp_codes = list(set(p["corp_code"] for p in all_purchases))
        logger.info(f"Fetching total shares for {len(unique_corp_codes)} companies...")

        async def fetch_total_shares_batch(corp_codes):
            results = {}
            for i in range(0, len(corp_codes), 10):
                batch = corp_codes[i:i+10]
                tasks = [get_total_shares(client, cc) for cc in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for cc, result in zip(batch, batch_results):
                    results[cc] = result if isinstance(result, int) else 0
                await asyncio.sleep(0.2)
            return results

        total_shares_map = await fetch_total_shares_batch(unique_corp_codes)

        for p in all_purchases:
            p["total_shares"] = total_shares_map.get(p["corp_code"], 0)

        # Sort by shares descending
        all_purchases.sort(key=lambda x: x["shares"], reverse=True)

        # Remove corp_code from response
        response_purchases = [{k: v for k, v in p.items() if k != "corp_code"} for p in all_purchases]

        stats = compute_stats(response_purchases)

        return {
            "success": True,
            "data": response_purchases,
            "stats": stats,
            "source": "dart_api",
            "message": f"총 {len(response_purchases)}건의 임원 자사주 매입 데이터를 조회했습니다."
        }


async def get_all_disclosures(client: httpx.AsyncClient, start_date: str, end_date: str) -> list:
    """Get all disclosure pages from DART API"""
    all_disclosures = []
    page_no = 1

    while True:
        url = f"{BASE_URL}/list.json"
        params = {
            "crtfc_key": API_KEY,
            "bgn_de": start_date,
            "end_de": end_date,
            "page_count": 100,
            "page_no": page_no
        }

        response = await client.get(url, params=params)
        data = response.json()

        if data.get("status") != "000":
            if data.get("status") == "013":
                break
            raise Exception(f"DART API Error: {data.get('message')}")

        all_disclosures.extend(data.get("list", []))

        total_page = data.get("total_page", 1)
        if page_no >= total_page:
            break

        page_no += 1
        await asyncio.sleep(0.1)

    return all_disclosures


async def get_executive_stock(client: httpx.AsyncClient, corp_code: str) -> list:
    """Get executive stock ownership for a company"""
    url = f"{BASE_URL}/elestock.json"
    params = {"crtfc_key": API_KEY, "corp_code": corp_code}
    response = await client.get(url, params=params)
    data = response.json()
    if data.get("status") == "000":
        return data.get("list", [])
    return []


# Cache for total shares data
total_shares_cache = {}

async def get_total_shares(client: httpx.AsyncClient, corp_code: str) -> int:
    """Get total issued shares for a company"""
    if corp_code in total_shares_cache:
        return total_shares_cache[corp_code]

    current_year = datetime.now().year
    report_codes = ["11011", "11014", "11012", "11013"]

    for year in [current_year, current_year - 1]:
        for reprt_code in report_codes:
            try:
                url = f"{BASE_URL}/stockTotqySttus.json"
                params = {
                    "crtfc_key": API_KEY,
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": reprt_code
                }
                response = await client.get(url, params=params)
                data = response.json()

                if data.get("status") == "000":
                    stock_list = data.get("list", [])
                    total = 0
                    for item in stock_list:
                        try:
                            istc_totqy = item.get("istc_totqy", "0")
                            total += int(str(istc_totqy).replace(",", "").replace("-", "0"))
                        except:
                            pass
                    if total > 0:
                        total_shares_cache[corp_code] = total
                        return total
            except:
                continue

    total_shares_cache[corp_code] = 0
    return 0


# Cache for report reasons
report_reason_cache = {}

async def get_report_reason(client: httpx.AsyncClient, rcept_no: str) -> str:
    """Get report reason from disclosure document"""
    if rcept_no in report_reason_cache:
        return report_reason_cache[rcept_no]

    try:
        url = f"{BASE_URL}/document.xml"
        params = {"crtfc_key": API_KEY, "rcept_no": rcept_no}
        response = await client.get(url, params=params)
        if response.status_code != 200:
            return ""

        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            for filename in zf.namelist():
                if filename.endswith('.xml'):
                    xml_content = zf.read(filename).decode('utf-8')
                    reasons = []

                    rpt_rsn_matches = re.findall(
                        r'AUNIT="RPT_RSN"[^>]*AUNITVALUE="(\d+)"[^>]*>([^<]*)',
                        xml_content
                    )
                    for code, text in rpt_rsn_matches:
                        reason_text = text.strip() if text.strip() else RPT_RSN_MAP.get(code, f"코드{code}")
                        if reason_text and reason_text not in reasons:
                            reasons.append(reason_text)

                    if not reasons:
                        chn_match = re.search(r'ACODE="CHN_RSN"[^>]*>([^<]+)', xml_content)
                        if chn_match:
                            reasons.append(chn_match.group(1).strip())

                    if reasons:
                        result = ", ".join(dict.fromkeys(reasons))
                        report_reason_cache[rcept_no] = result
                        return result

        report_reason_cache[rcept_no] = ""
        return ""
    except Exception as e:
        logger.error(f"Error fetching report reason for {rcept_no}: {e}")
        return ""


@app.get("/api/report-reason")
async def get_report_reason_api(rcept_no: str = Query(..., description="Report number")):
    """Get the reason for stock change from disclosure document"""
    # Check DB first
    try:
        sb = get_supabase()
        result = sb.table("executive_purchases") \
            .select("reason") \
            .eq("rcept_no", rcept_no) \
            .not_.is_("reason", "null") \
            .neq("reason", "") \
            .limit(1) \
            .execute()
        if result.data and result.data[0].get("reason"):
            return {"rcept_no": rcept_no, "reason": result.data[0]["reason"]}
    except:
        pass

    # Fallback to DART API
    async with httpx.AsyncClient(timeout=30.0) as client:
        reason = await get_report_reason(client, rcept_no)

        # Store reason back to DB
        if reason:
            try:
                sb = get_supabase()
                sb.table("executive_purchases") \
                    .update({"reason": reason}) \
                    .eq("rcept_no", rcept_no) \
                    .execute()
            except:
                pass

        return {"rcept_no": rcept_no, "reason": reason}


@app.get("/api/cron/update")
async def cron_update(
    secret: str = Query("", description="Cron secret for auth"),
    date: Optional[str] = Query(None, description="Specific date YYYYMMDD (default: today)")
):
    """
    Daily cron endpoint to update today's data.
    Called by Vercel Cron or manually with ?secret=xxx
    """
    # Verify cron secret
    auth_header = ""
    if not secret:
        # Also check Authorization header (Vercel sends this)
        return {"success": False, "message": "Unauthorized"}

    if secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Invalid cron secret")

    # Determine target date
    if date:
        target_date = date
    else:
        target_date = datetime.now().strftime("%Y%m%d")

    target_dt = datetime.strptime(target_date, "%Y%m%d").date()

    logger.info(f"Cron update started for {target_date}")

    try:
        sb = get_supabase()

        # Mark as in_progress
        sb.table("update_log").upsert(
            {"update_date": target_dt.isoformat(), "status": "in_progress", "records_count": 0},
            on_conflict="update_date"
        ).execute()

        # Fetch from DART API
        result = await fetch_from_dart(target_date, target_date)

        if result["success"]:
            records = result["data"]

            if records:
                # Store to DB
                await store_to_db(records, target_dt, target_dt)

            # Update log
            sb.table("update_log").upsert(
                {
                    "update_date": target_dt.isoformat(),
                    "records_count": len(records),
                    "status": "success"
                },
                on_conflict="update_date"
            ).execute()

            logger.info(f"Cron update complete: {len(records)} records for {target_date}")
            return {
                "success": True,
                "date": target_date,
                "records": len(records),
                "message": f"{len(records)}건 업데이트 완료"
            }
        else:
            raise Exception("DART API fetch failed")

    except Exception as e:
        logger.error(f"Cron update error: {e}")
        try:
            sb = get_supabase()
            sb.table("update_log").upsert(
                {
                    "update_date": target_dt.isoformat(),
                    "status": "failed",
                    "error_message": str(e)
                },
                on_conflict="update_date"
            ).execute()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/update-status")
async def get_update_status():
    """Get the latest update status from DB"""
    try:
        sb = get_supabase()
        result = sb.table("update_log") \
            .select("*") \
            .order("update_date", desc=True) \
            .limit(10) \
            .execute()
        return {"success": True, "data": result.data}
    except Exception as e:
        return {"success": False, "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
