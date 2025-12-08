"""
Vercel Serverless Function for DART API
Single file with multiple routes to work within Vercel's constraints
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "9361d74facc8c239f634b08c0f436192de5c14de"
BASE_URL = "https://opendart.fss.or.kr/api"


@app.get("/")
@app.get("/api/disclosures")
async def get_disclosures(
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)")
):
    """Get disclosure list and return unique companies with executive reports"""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
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

            keywords = ["임원", "주요주주", "소유상황"]
            unique_corps = {}

            for d in all_disclosures:
                report_nm = d.get("report_nm", "")
                if any(kw in report_nm for kw in keywords):
                    corp_code = d.get("corp_code")
                    if corp_code and corp_code not in unique_corps:
                        unique_corps[corp_code] = {
                            "corp_code": corp_code,
                            "corp_name": d.get("corp_name"),
                            "corp_cls": d.get("corp_cls"),
                            "stock_code": d.get("stock_code", "")
                        }

            return {
                "success": True,
                "companies": list(unique_corps.values()),
                "total_disclosures": len(all_disclosures),
                "filtered_companies": len(unique_corps)
            }

    except Exception as e:
        return {"success": False, "companies": [], "message": str(e)}


@app.get("/api/executive")
async def get_executive(
    corp_code: str = Query(..., description="Company code"),
    corp_name: str = Query("", description="Company name"),
    corp_cls: str = Query("", description="Market type (Y/K/N/E)"),
    stock_code: str = Query("", description="Stock code")
):
    """Get executive stock ownership for a single company"""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            url = f"{BASE_URL}/elestock.json"
            params = {
                "crtfc_key": API_KEY,
                "corp_code": corp_code
            }

            response = await client.get(url, params=params)
            data = response.json()

            purchases = []

            if data.get("status") == "000":
                stock_list = data.get("list", [])

                for item in stock_list:
                    is_exec = item.get("isu_exctv_rgist_at", "") in ["등록", "등기임원", "비등기임원"]
                    ofcps = item.get("isu_exctv_ofcps", "")
                    has_position = ofcps and ofcps != "-" and len(ofcps) > 0

                    irds_cnt_str = item.get("sp_stock_lmp_irds_cnt", "0")
                    try:
                        irds_cnt = int(str(irds_cnt_str).replace(",", "").replace("-", "0"))
                    except:
                        irds_cnt = 0

                    if (is_exec or has_position) and irds_cnt > 0:
                        market = corp_cls
                        if market == "Y":
                            market = "코스피"
                        elif market == "K":
                            market = "코스닥"
                        else:
                            market = "기타"

                        purchases.append({
                            "corp_name": corp_name,
                            "stock_code": stock_code,
                            "market": market,
                            "exec_name": item.get("repror", ""),
                            "position": ofcps,
                            "shares": irds_cnt,
                            "rate": item.get("sp_stock_lmp_rate", "")
                        })

            return {
                "success": True,
                "corp_code": corp_code,
                "purchases": purchases
            }

    except Exception as e:
        return {"success": False, "corp_code": corp_code, "purchases": [], "message": str(e)}
