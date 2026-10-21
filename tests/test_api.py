from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

import server
from conftest import make_document, trade_row, zip_document

LIST_ROWS = [
    {"corp_code": "00000001", "corp_name": "가나전자", "stock_code": "000001", "corp_cls": "Y",
     "report_nm": "임원ㆍ주요주주특정증권등소유상황보고서", "rcept_no": "20261021000001", "flr_nm": "홍길동",
     "rcept_dt": "20261021", "rm": ""},
    {"corp_code": "00000002", "corp_name": "다라바이오", "stock_code": "000002", "corp_cls": "K",
     "report_nm": "임원ㆍ주요주주특정증권등소유상황보고서", "rcept_no": "20261021000002", "flr_nm": "김철수",
     "rcept_dt": "20261021", "rm": ""},
]

DOCS = {
    "20261021000001": make_document([trade_row("장내매수(+)", "20261020", "1,000", "10,000")]),
    "20261021000002": make_document([trade_row("주식매수선택권(+)", "20261020", "500", "5,000", code="58")]),
}


@pytest.fixture
def api(monkeypatch):
    calls = []
    state = {"list": {"status": "000", "total_page": 1, "list": LIST_ROWS}, "doc_status": None}

    def handler(request):
        calls.append(request.url.path)
        if "siseJson" not in request.url.path:
            assert request.url.params["crtfc_key"] == "test-key"
        if request.url.path.endswith("/list.json"):
            return httpx.Response(200, json=state["list"])
        if request.url.path.endswith("/document.xml"):
            if state["doc_status"]:
                return httpx.Response(200, json={"status": state["doc_status"], "message": "error"})
            return httpx.Response(200, content=zip_document(DOCS[request.url.params["rcept_no"]]))
        if "siseJson" in request.url.path:
            return httpx.Response(200, text="""[['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
["20261020", 10000, 10500, 9900, 10200, 12345, 1.5],
["20261021", 10200, 10300, 10000, 10100, 23456, 1.5]]""")
        return httpx.Response(404)

    monkeypatch.setattr(server, "DART_API_KEY", "test-key")
    monkeypatch.setattr(server, "new_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(server, "dart_throttle", server.Throttle(10_000))
    monkeypatch.setattr(server, "naver_throttle", server.Throttle(10_000))
    monkeypatch.setattr(server, "dart_budget", server.DailyBudget(1_000))
    for name in ("documents", "listings", "days", "prices"):
        monkeypatch.setattr(server, name, server.TTLCache(100))
    monkeypatch.setattr(server, "today_kst", lambda: date(2026, 10, 22))
    return TestClient(server.app), calls, state


def test_day_endpoint_returns_only_trades(api):
    client, calls, _ = api
    res = client.get("/api/filings/20261021")
    assert res.status_code == 200
    body = res.json()
    assert body["complete"] is True
    assert body["summary"] == {"total": 2, "trades": 1, "other": 1, "failed": 0, "pending": 0}
    [record] = body["filings"]
    assert record["corp_name"] == "가나전자"
    assert record["market"] == "코스피"
    assert record["buy"]["amount"] == 10_000_000
    assert "s-maxage=1800" in res.headers["cache-control"]  # recent day: amendments may still come

    calls.clear()
    assert client.get("/api/filings/20261021").json() == body
    assert calls == []  # served from memory


def test_empty_day(api):
    client, _, state = api
    state["list"] = {"status": "013", "message": "조회된 데이타가 없습니다."}
    res = client.get("/api/filings/20261001")
    assert res.json()["filings"] == [] and res.json()["complete"] is True
    assert "s-maxage=21600" in res.headers["cache-control"]


def test_unparseable_document_is_reported_not_cached(api):
    client, calls, _ = api
    DOCS["20261021000002"], original = "<DOCUMENT>broken</DOCUMENT>", DOCS["20261021000002"]
    try:
        res = client.get("/api/filings/20261021")
        body = res.json()
        assert res.status_code == 200
        assert body["complete"] is False
        assert body["failed"] == [{"rcept_no": "20261021000002", "corp_name": "다라바이오", "filer": "김철수"}]
        assert len(body["filings"]) == 1
        assert res.headers["cache-control"] == "no-store"
    finally:
        DOCS["20261021000002"] = original
    calls.clear()
    assert client.get("/api/filings/20261021").json()["complete"] is True
    assert calls.count("/api/document.xml") == 1  # only the failed one is fetched again


def test_slow_documents_come_back_as_pending(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr(server, "DAY_TIME_BUDGET", 0.0)
    body = client.get("/api/filings/20261021").json()
    assert body["complete"] is False
    assert sorted(body["pending"]) == ["20261021000001", "20261021000002"]


@pytest.mark.parametrize("status, http_status", [("020", 429), ("010", 503), ("012", 502), ("800", 503)])
def test_dart_status_codes_become_clear_errors(api, status, http_status):
    client, _, state = api
    state["list"] = {"status": status, "message": "error"}
    res = client.get("/api/filings/20261021")
    assert res.status_code == http_status
    assert res.headers["cache-control"] == "no-store"
    assert res.json()["detail"]


def test_fatal_document_error_aborts_the_day(api):
    client, _, state = api
    state["doc_status"] = "020"
    assert client.get("/api/filings/20261021").status_code == 429


def test_missing_key(api, monkeypatch):
    client, calls, _ = api
    monkeypatch.setattr(server, "DART_API_KEY", "")
    res = client.get("/api/filings/20261021")
    assert res.status_code == 503
    assert "DART_API_KEY" in res.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("value", ["2026-10-21", "20261032", "20261023", "20240101", "abc"])
def test_rejects_bad_dates(api, value):
    client, calls, _ = api
    assert client.get(f"/api/filings/{value}").status_code in (400, 404)
    assert calls == []


def test_prices(api):
    client, _, _ = api
    res = client.get("/api/prices/000001?period=3m")
    assert res.status_code == 200
    prices = res.json()["prices"]
    assert prices[0] == {"date": "2026-10-20", "open": 10000.0, "high": 10500.0, "low": 9900.0,
                         "close": 10200.0, "volume": 12345}
    assert client.get("/api/prices/00000A?period=3m").status_code == 200
    assert client.get("/api/prices/../etc?period=3m").status_code == 404
    assert client.get("/api/prices/1234567?period=3m").status_code == 400
    assert client.get("/api/prices/000001?period=10y").status_code == 400


def test_index_has_security_headers(api):
    client, _, _ = api
    res = client.get("/")
    assert res.status_code == 200
    assert "frame-ancestors 'none'" in res.headers["content-security-policy"]
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "crtfc_key" not in res.text


def test_lookback_window(api):
    client, _, _ = api
    oldest = date(2026, 10, 22) - timedelta(days=server.MAX_LOOKBACK_DAYS)
    assert client.get(f"/api/filings/{oldest:%Y%m%d}").status_code == 200
    assert client.get(f"/api/filings/{oldest - timedelta(days=1):%Y%m%d}").status_code == 400
