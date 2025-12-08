"""
Vercel Serverless Function for DART API
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

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


@app.get("/api/purchases")
async def get_purchases(
    start_date: str = Query(..., description="Start date (YYYYMMDD)"),
    end_date: str = Query(..., description="End date (YYYYMMDD)")
):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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

            # Step 3: Get executive stock ownership for each company
            all_purchases = []

            for i, (corp_code, corp_info) in enumerate(unique_corps.items()):
                try:
                    stock_list = await get_executive_stock(client, corp_code)

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
                            market = corp_info["corp_cls"]
                            if market == "Y":
                                market = "코스피"
                            elif market == "K":
                                market = "코스닥"
                            else:
                                market = "기타"

                            all_purchases.append({
                                "corp_name": corp_info["corp_name"],
                                "stock_code": corp_info["stock_code"],
                                "market": market,
                                "exec_name": item.get("repror", ""),
                                "position": ofcps,
                                "shares": irds_cnt,
                                "rate": item.get("sp_stock_lmp_rate", "")
                            })

                    if (i + 1) % 50 == 0:
                        await asyncio.sleep(0.3)

                except Exception as e:
                    print(f"Error fetching {corp_code}: {e}")
                    continue

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

            return {
                "success": True,
                "data": all_purchases,
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

    except Exception as e:
        return {"success": False, "data": [], "stats": {}, "message": str(e)}


async def get_all_disclosures(client: httpx.AsyncClient, start_date: str, end_date: str) -> list:
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
