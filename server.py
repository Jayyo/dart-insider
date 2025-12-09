"""
DART Insider Web Server
FastAPI backend with CORS support for DART API proxy
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import logging
import time
import hashlib
import json
import zipfile
import io
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# In-memory cache for API responses
# Structure: {cache_key: {"data": response_data, "timestamp": datetime, "stats": stats}}
api_cache = {}
CACHE_TTL_MINUTES = 30  # Cache expires after 30 minutes

def get_cache_key(start_date: str, end_date: str) -> str:
    """Generate cache key from date range"""
    return f"{start_date}_{end_date}"

def is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry exists and is not expired"""
    if cache_key not in api_cache:
        return False
    cache_entry = api_cache[cache_key]
    age = datetime.now() - cache_entry["timestamp"]
    return age < timedelta(minutes=CACHE_TTL_MINUTES)

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

    # Get client info
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    real_ip = forwarded_for.split(",")[0].strip() if forwarded_for else client_ip

    user_agent = request.headers.get("User-Agent", "unknown")
    method = request.method
    path = request.url.path
    query = str(request.query_params) if request.query_params else ""

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = round((time.time() - start_time) * 1000, 2)

    # Log the request
    logger.info(
        f"IP: {real_ip} | {method} {path} | Query: {query} | "
        f"Status: {response.status_code} | Time: {duration}ms | UA: {user_agent[:80]}"
    )

    return response

# Configuration
API_KEY = "9361d74facc8c239f634b08c0f436192de5c14de"
BASE_URL = "https://opendart.fss.or.kr/api"

# Response models
class ExecutivePurchase(BaseModel):
    corp_name: str
    stock_code: str
    market: str
    exec_name: str
    position: str
    shares: int
    rate: str
    report_date: str = ""  # YYYY-MM-DD format
    total_shares: int = 0  # Total issued shares of the company

class PurchaseResponse(BaseModel):
    success: bool
    data: list[ExecutivePurchase]
    stats: dict
    message: str = ""


@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")


@app.get("/api/purchases")
async def get_purchases(
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)"),
    force_refresh: bool = Query(False, description="Force refresh cache")
):
    """
    Get executive stock purchases for a date range.
    This endpoint handles all the DART API calls and data processing.
    Only returns data where the report date (rcept_dt) is within the query range.
    """
    try:
        # Check cache first
        cache_key = get_cache_key(start_date, end_date)
        if not force_refresh and is_cache_valid(cache_key):
            cached = api_cache[cache_key]
            logger.info(f"Cache HIT for {start_date}-{end_date} (age: {(datetime.now() - cached['timestamp']).seconds}s)")
            return cached["data"]

        logger.info(f"Cache MISS for {start_date}-{end_date}, fetching from DART API...")

        # Convert dates for comparison (YYYYMMDD format)
        start_dt = start_date  # Already in YYYYMMDD format
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

            logger.info(f"Found {len(unique_corps)} companies with executive reports in date range {start_date}-{end_date}")

            # Step 3: Get executive stock ownership for each company
            all_purchases = []

            for i, (corp_code, corp_info) in enumerate(unique_corps.items()):
                try:
                    stock_list = await get_executive_stock(client, corp_code)

                    for item in stock_list:
                        # Filter by report date - only include if within query range
                        rcept_dt = item.get("rcept_dt", "").replace("-", "")
                        if not rcept_dt or rcept_dt < start_dt or rcept_dt > end_dt:
                            continue  # Skip if outside date range

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

                    # Rate limiting
                    if (i + 1) % 50 == 0:
                        await asyncio.sleep(0.3)

                except Exception as e:
                    logger.error(f"Error fetching {corp_code}: {e}")
                    continue

            # Fetch total shares for unique companies in parallel
            unique_corp_codes = list(set(p["corp_code"] for p in all_purchases))
            logger.info(f"Fetching total shares for {len(unique_corp_codes)} unique companies...")

            # Batch fetch total shares (limit concurrent requests)
            async def fetch_total_shares_batch(corp_codes):
                results = {}
                for i in range(0, len(corp_codes), 10):  # Process 10 at a time
                    batch = corp_codes[i:i+10]
                    tasks = [get_total_shares(client, cc) for cc in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for cc, result in zip(batch, batch_results):
                        results[cc] = result if isinstance(result, int) else 0
                    await asyncio.sleep(0.2)  # Rate limiting between batches
                return results

            total_shares_map = await fetch_total_shares_batch(unique_corp_codes)

            # Update purchases with total shares
            for p in all_purchases:
                p["total_shares"] = total_shares_map.get(p["corp_code"], 0)

            # Sort by shares descending
            all_purchases.sort(key=lambda x: x["shares"], reverse=True)

            # Calculate stats
            companies = len(set(p["corp_name"] for p in all_purchases))
            executives = len(all_purchases)
            total_shares = sum(p["shares"] for p in all_purchases)

            # Find top company
            company_shares = {}
            for p in all_purchases:
                company_shares[p["corp_name"]] = company_shares.get(p["corp_name"], 0) + p["shares"]

            top_company = ""
            top_company_shares = 0
            if company_shares:
                top_company = max(company_shares, key=company_shares.get)
                top_company_shares = company_shares[top_company]

            # Market distribution
            market_dist = {"코스피": 0, "코스닥": 0, "기타": 0}
            for p in all_purchases:
                if p["market"] in market_dist:
                    market_dist[p["market"]] += p["shares"]

            logger.info(f"Returning {len(all_purchases)} purchases for date range {start_date}-{end_date}")

            # Remove corp_code from response (internal use only)
            response_purchases = [{k: v for k, v in p.items() if k != "corp_code"} for p in all_purchases]

            response_data = {
                "success": True,
                "data": response_purchases,
                "stats": {
                    "companies": companies,
                    "executives": executives,
                    "total_shares": total_shares,
                    "top_company": top_company,
                    "top_company_shares": top_company_shares,
                    "market_distribution": market_dist
                },
                "message": f"총 {len(all_purchases)}건의 임원 자사주 매입 데이터를 조회했습니다."
            }

            # Store in cache
            api_cache[cache_key] = {
                "data": response_data,
                "timestamp": datetime.now()
            }
            logger.info(f"Cached response for {start_date}-{end_date}")

            return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            if data.get("status") == "013":  # No data
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
    params = {
        "crtfc_key": API_KEY,
        "corp_code": corp_code
    }

    response = await client.get(url, params=params)
    data = response.json()

    if data.get("status") == "000":
        return data.get("list", [])
    return []


# Cache for total shares data (company -> total_shares)
total_shares_cache = {}

async def get_total_shares(client: httpx.AsyncClient, corp_code: str) -> int:
    """Get total issued shares for a company from latest annual report"""
    # Check cache first
    if corp_code in total_shares_cache:
        return total_shares_cache[corp_code]

    # Try recent years and report codes
    current_year = datetime.now().year
    report_codes = ["11011", "11014", "11012", "11013"]  # Annual, Q3, Semi, Q1

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
                    # Sum up all stock types (common + preferred)
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


# Cache for report reasons (rcept_no -> reason text)
report_reason_cache = {}

async def get_report_reason(client: httpx.AsyncClient, rcept_no: str) -> str:
    """Get report reason from disclosure document"""
    # Check cache first
    if rcept_no in report_reason_cache:
        return report_reason_cache[rcept_no]

    try:
        url = f"{BASE_URL}/document.xml"
        params = {
            "crtfc_key": API_KEY,
            "rcept_no": rcept_no
        }

        response = await client.get(url, params=params)
        if response.status_code != 200:
            return ""

        # Unzip and parse XML
        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            for filename in zf.namelist():
                if filename.endswith('.xml'):
                    xml_content = zf.read(filename).decode('utf-8')
                    # Extract change reason using regex (CHN_RSN = 변동사유)
                    match = re.search(r'ACODE="CHN_RSN"[^>]*>([^<]+)', xml_content)
                    if match:
                        reason = match.group(1).strip()
                        report_reason_cache[rcept_no] = reason
                        return reason

        report_reason_cache[rcept_no] = ""
        return ""
    except Exception as e:
        logger.error(f"Error fetching report reason for {rcept_no}: {e}")
        return ""


@app.get("/api/report-reason")
async def get_report_reason_api(rcept_no: str = Query(..., description="Report number")):
    """Get the reason for stock change from disclosure document"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        reason = await get_report_reason(client, rcept_no)
        return {"rcept_no": rcept_no, "reason": reason}


# Mount static files (for any additional assets)
# app.mount("/static", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
