"""Parsing for OpenDART 임원ㆍ주요주주특정증권등소유상황보고서 filings.

The disclosure XML is not strict XML, so cells are picked out by their
ACODE / AUNIT attributes with regular expressions instead of a parser.
"""

import html
import io
import re
import zipfile
import zlib
from decimal import Decimal, InvalidOperation

MARKETS = {"Y": "코스피", "K": "코스닥", "N": "코넥스", "E": "기타"}
STOCK_KINDS = ("보통주", "우선주")

MAX_ZIP_BYTES = 5 * 1024 * 1024
MAX_XML_BYTES = 10 * 1024 * 1024

_CELL = re.compile(r"<(TE|TU|TD)\b([^>]*)>(.*?)</\1>", re.S)
_ROW = re.compile(r"<TR\b[^>]*>(.*?)</TR>", re.S)
_CODE = re.compile(r'\b(?:ACODE|AUNIT)="([A-Z0-9_]+)"')
_UNIT_VALUE = re.compile(r'\bAUNITVALUE="([^"]*)"')
_TAG = re.compile(r"<[^>]+>")
_TRADE_TABLE = re.compile(r'<TABLE-GROUP\b[^>]*ACLASS="RPT_RSN"[^>]*>(.*?)</TABLE-GROUP>', re.S)
_SIGN_SUFFIX = re.compile(r"\(\s*[+-]\s*\)$")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_FOOTNOTE = re.compile(r"\(?\s*(?:주|\*|※)\s*\d+\s*\)?")


class DocumentError(ValueError):
    pass


def clean_text(fragment):
    text = html.unescape(_TAG.sub("", fragment or ""))
    return re.sub(r"\s+", " ", text).strip()


def to_int(value):
    """'1,000' -> 1000, '-1,000' / '△1,000' / '(1,000)' -> -1000, '-' or '' -> None."""
    text = clean_text(value).replace(",", "").replace(" ", "")
    text = text.replace("△", "-").replace("−", "-")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    match = re.fullmatch(r"(-?\d+)주?", text)
    return int(match.group(1)) if match else None


def to_price(value):
    """Unit price from cells like '160,100', '5,000 (원)', '취득(3,975원)' or '-/24,000'.

    A cell with two different numbers (a price range, for example) is left as
    unknown instead of guessing which one is the price.
    """
    numbers = set()
    for token in _NUMBER.findall(_FOOTNOTE.sub(" ", clean_text(value))):
        token = token.rstrip(",")
        if "," in token and not _GROUPED.fullmatch(token):
            return None
        try:
            numbers.add(Decimal(token.replace(",", "")))
        except InvalidOperation:
            return None
    if len(numbers) != 1:
        return None
    price = numbers.pop()
    return price if price > 0 else None


def to_date(unit_value, text):
    """Prefer the machine value (YYYYMMDD); fall back to the visible text."""
    for candidate in (unit_value or "", clean_text(text)):
        digits = re.findall(r"\d+", candidate)
        if len(digits) == 1 and len(digits[0]) == 8:
            y, m, d = digits[0][:4], digits[0][4:6], digits[0][6:]
        elif len(digits) >= 3 and len(digits[0]) == 4:
            y, m, d = digits[0], digits[1].zfill(2), digits[2].zfill(2)
        else:
            continue
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{m}-{d}"
    return None


def _cells(fragment):
    """Yield (code, unit_value, raw_text) for every coded cell in a fragment."""
    for _tag, attrs, inner in _CELL.findall(fragment):
        code = _CODE.search(attrs)
        if not code:
            continue
        unit = _UNIT_VALUE.search(attrs)
        yield code.group(1), unit.group(1) if unit else None, inner


def _header(xml):
    values = {}
    for code, unit, inner in _cells(xml):
        values.setdefault(code, (unit, inner))
    return values


def classify(reason, shares):
    """Return (kind, venue) for one change row.

    Only open-market, off-market and after-hours trades count as trades.
    Option exercises, rights issues, gifts, splits, mergers, listings and
    new appointments are reported as 'other'.
    """
    base = _SIGN_SUFFIX.sub("", re.sub(r"\s+", "", reason or ""))
    if not shares:
        return "other", None
    if base.startswith("장내매수") and shares > 0:
        return "buy", "장내"
    if base.startswith("장외매수") and shares > 0:
        return "buy", "장외"
    if base.startswith("장내매도") and shares < 0:
        return "sell", "장내"
    if base.startswith("장외매도") and shares < 0:
        return "sell", "장외"
    if base.startswith("시간외매매"):
        return ("buy" if shares > 0 else "sell"), "시간외"
    return "other", None


def parse_trades(xml):
    table = _TRADE_TABLE.search(xml)
    if not table:
        raise DocumentError("세부변동내역 table not found")
    trades = []
    for row in _ROW.findall(table.group(1)):
        cells = {code: (unit, inner) for code, unit, inner in _cells(row)}
        if "RPT_RSN" not in cells or "MDF_STK_CNT" not in cells:
            continue  # header or total row
        reason = clean_text(cells["RPT_RSN"][1])
        shares = to_int(cells["MDF_STK_CNT"][1])
        price_cell = next((v for k, v in cells.items() if k.startswith("ACI_AMT") and "SUM" not in k), None)
        plan = cells.get("TRD_PLN_RD")
        security = clean_text(cells["STR_KND"][1]) if "STR_KND" in cells else ""
        kind, venue = classify(reason, shares)
        if kind != "other" and security not in STOCK_KINDS:
            kind, venue = "other", None
        trades.append({
            "date": to_date(*cells["MDF_DM"]) if "MDF_DM" in cells else None,
            "reason": _SIGN_SUFFIX.sub("", reason).strip(),
            "kind": kind,
            "venue": venue,
            "security": security,
            "shares": shares,
            "price": to_price(price_cell[1]) if price_cell else None,
            "before": to_int(cells["BFR_STK_CNT"][1]) if "BFR_STK_CNT" in cells else None,
            "after": to_int(cells["AFR_STK_CNT"][1]) if "AFR_STK_CNT" in cells else None,
            "plan_date": to_date(*plan) if plan else None,
        })
    return trades


def _side(trades, kind):
    rows = [t for t in trades if t["kind"] == kind]
    shares = sum(abs(t["shares"]) for t in rows)
    priced = [t for t in rows if t["price"] is not None]
    priced_shares = sum(abs(t["shares"]) for t in priced)
    amount = sum((abs(t["shares"]) * t["price"] for t in priced), Decimal(0))
    dates = sorted(t["date"] for t in rows if t["date"])
    return {
        "shares": shares,
        "amount": int(amount.to_integral_value()) if priced else None,
        "priced_shares": priced_shares,
        "avg_price": float(round(amount / priced_shares, 2)) if priced_shares else None,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "venues": sorted({t["venue"] for t in rows}),
        "securities": sorted({t["security"] for t in rows}),
    }


def parse_document(xml):
    header = _header(xml)

    def value(code):
        return clean_text(header[code][1]) if code in header else ""

    def unit(code):
        return header[code][0] if code in header else None

    trades = parse_trades(xml)
    registered = value("STF_RYN")
    plan_dates = sorted({t["plan_date"] for t in trades if t["plan_date"]})
    return {
        "reporter": value("IFR_NM").replace(" ", "") if re.fullmatch(r"[가-힣 ]+", value("IFR_NM")) else value("IFR_NM"),
        "reporter_type": value("IFR_TP"),
        "registered": registered if registered not in ("", "-") else None,
        "is_executive": unit("STF_RYN") in ("Y", "N") or registered in ("등기임원", "비등기임원"),
        "position": value("STF_PSM") if value("STF_PSM") not in ("", "-") else None,
        "major_holder": value("MAIN_SH") if value("MAIN_SH") not in ("", "-") else None,
        "ownership_before": _ratio(value("BFR_PS_CPT_RT")),
        "ownership_after": _ratio(value("AFR_PS_CPT_RT") or value("HLD_STK_RT")),
        "plan_dates": plan_dates,
        "trades": trades,
        "buy": _side(trades, "buy"),
        "sell": _side(trades, "sell"),
    }


def _ratio(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def read_document_zip(content):
    """Return the main XML text from the document.xml zip, with size limits."""
    if len(content) > MAX_ZIP_BYTES:
        raise DocumentError("document archive too large")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [i for i in archive.infolist() if i.filename.lower().endswith(".xml")]
            if not names or len(archive.infolist()) > 20:
                raise DocumentError("unexpected archive contents")
            info = max(names, key=lambda i: i.file_size)
            if info.file_size > MAX_XML_BYTES:
                raise DocumentError("document too large")
            raw = archive.read(info)
    except DocumentError:
        raise
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as exc:
        raise DocumentError(f"broken document archive: {exc}") from exc
    head = raw[:200].decode("ascii", "ignore").lower()
    encoding = "euc-kr" if "euc-kr" in head else "utf-8"
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        return raw.decode("cp949", "replace")


def _number(value):
    if value is None:
        return None
    return int(value) if value == value.to_integral_value() else float(value)


def public_record(item, doc):
    """Merge a list.json row and a parsed document into the API shape."""
    return {
        "rcept_no": item["rcept_no"],
        "report_date": item["report_date"],
        "corp_code": item["corp_code"],
        "corp_name": item["corp_name"],
        "stock_code": item["stock_code"],
        "market": item["market"],
        "amended": item["amended"],
        "reporter": item["filer"] or doc["reporter"],
        "reporter_type": doc["reporter_type"],
        "is_executive": doc["is_executive"],
        "registered": doc["registered"],
        "position": doc["position"],
        "major_holder": doc["major_holder"],
        "ownership_before": doc["ownership_before"],
        "ownership_after": doc["ownership_after"],
        "plan_dates": doc["plan_dates"],
        "buy": doc["buy"],
        "sell": doc["sell"],
        "trades": [{**t, "price": _number(t["price"])} for t in doc["trades"]],
    }


def list_item(item):
    """Normalize one list.json row."""
    rcept_dt = item.get("rcept_dt") or ""
    return {
        "rcept_no": item.get("rcept_no", ""),
        "report_date": f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}" if len(rcept_dt) == 8 else None,
        "corp_code": item.get("corp_code", ""),
        "corp_name": item.get("corp_name", ""),
        "stock_code": (item.get("stock_code") or "").strip(),
        "market": MARKETS.get(item.get("corp_cls", ""), "기타"),
        "filer": item.get("flr_nm", ""),
        "amended": "정정" in (item.get("report_nm") or ""),
    }
