"""
Vercel Serverless Function - Get executive stock info for a single company
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
