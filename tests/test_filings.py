from decimal import Decimal

import pytest

import filings
from conftest import make_document, trade_row, zip_document


def test_on_market_buys_are_summed_with_disclosed_prices():
    doc = filings.parse_document(make_document([
        trade_row("장내매수(+)", "20261012", "50,000", "12,078", plan="20260828"),
        trade_row("장내매수(+)", "20261013", "40,000", "12,168", plan="20260828"),
    ]))

    assert doc["reporter"] == "홍길동"
    assert doc["is_executive"] is True
    assert doc["registered"] == "등기임원"
    assert doc["position"] == "대표이사"
    assert doc["ownership_before"] == 1.25
    assert doc["ownership_after"] == 1.5
    assert doc["plan_dates"] == ["2026-08-28"]

    buy = doc["buy"]
    assert buy["shares"] == 90_000
    assert buy["amount"] == 50_000 * 12_078 + 40_000 * 12_168
    assert buy["avg_price"] == pytest.approx(buy["amount"] / 90_000, abs=0.01)
    assert (buy["first_date"], buy["last_date"]) == ("2026-10-12", "2026-10-13")
    assert buy["venues"] == ["장내"]
    assert doc["sell"]["shares"] == 0


def test_negative_changes_are_not_read_as_purchases():
    # The original bug turned "-12,687,372" into a 12,687,372-share purchase.
    doc = filings.parse_document(make_document([
        trade_row("주식병합", "20260515", '<SPAN USERMARK="0X3D3D3D">-12,687,372</SPAN>', "-", code="89",
                  before="<SPAN>14,097,079</SPAN>", after="1,409,707"),
        trade_row("시간외매매(-)", "20261019", "-173,300", "17,250", code="82"),
    ]))

    kinds = [(t["reason"], t["kind"], t["shares"]) for t in doc["trades"]]
    assert kinds == [("주식병합", "other", -12_687_372), ("시간외매매", "sell", -173_300)]
    assert doc["buy"]["shares"] == 0
    assert doc["sell"]["shares"] == 173_300
    assert doc["sell"]["amount"] == 173_300 * 17_250
    assert doc["sell"]["venues"] == ["시간외"]


@pytest.mark.parametrize("reason", ["주식매수선택권(+)", "신규상장(+)", "합병(+)", "증여(-)", "유상신주취득(+)", "신규선임(+)"])
def test_non_trade_reasons_are_excluded(reason):
    change = "-1,000" if reason.endswith("(-)") else "1,000"
    doc = filings.parse_document(make_document([trade_row(reason, "20261019", change, "5,000 (원)", code="58")]))

    assert doc["trades"][0]["kind"] == "other"
    assert doc["buy"]["shares"] == 0 and doc["sell"]["shares"] == 0


def test_mixed_filing_keeps_both_sides():
    doc = filings.parse_document(make_document([
        trade_row("장내매수(+)", "20261001", "500", "10,000"),
        trade_row("장내매도(-)", "20261008", "-200", "12,000"),
        trade_row("장외매수(+)", "20261010", "1,000", "9,500"),
    ]))

    assert doc["buy"]["shares"] == 1_500
    assert doc["buy"]["venues"] == ["장내", "장외"]
    assert doc["sell"]["shares"] == 200
    assert doc["sell"]["amount"] == 2_400_000


def test_missing_prices_are_unknown_not_zero():
    doc = filings.parse_document(make_document([
        trade_row("장내매수(+)", "20261001", "300", "-"),
        trade_row("장내매수(+)", "20261002", "100", "10,000"),
    ]))
    buy = doc["buy"]
    assert buy["shares"] == 400
    assert buy["priced_shares"] == 100
    assert buy["amount"] == 1_000_000

    only_missing = filings.parse_document(make_document([trade_row("장내매수(+)", "20261001", "300", "-")]))
    assert only_missing["buy"]["amount"] is None
    assert only_missing["buy"]["avg_price"] is None


def test_non_stock_securities_are_not_counted_as_share_trades():
    doc = filings.parse_document(make_document([
        trade_row("장외매도(-)", "20250317", "-20,000,000", "100", security="전환사채권"),
    ]))
    assert doc["trades"][0]["kind"] == "other"
    assert doc["sell"]["shares"] == 0


def test_major_shareholder_who_is_not_an_executive():
    doc = filings.parse_document(make_document(
        [trade_row("장내매수(+)", "20261020", "78,080", "63,882")],
        name="에이비씨홀딩스", ryn=("98", "-"), position="-", main=("1", "10%이상주주"), ifr_tp=("I", "국내법인"),
    ))
    assert doc["is_executive"] is False
    assert doc["registered"] is None
    assert doc["position"] is None
    assert doc["major_holder"] == "10%이상주주"
    assert doc["reporter_type"] == "국내법인"


def test_missing_trade_table_raises():
    with pytest.raises(filings.DocumentError):
        filings.parse_document("<DOCUMENT><P>no tables</P></DOCUMENT>")


@pytest.mark.parametrize("raw, expected", [
    ("1,000", 1000), ("-1,000", -1000), ("△1,000", -1000), ("(1,000)", -1000),
    ("<SPAN>2,500</SPAN>", 2500), ("-", None), ("", None), ("1,000 주", 1000),
])
def test_to_int(raw, expected):
    assert filings.to_int(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("160,100", Decimal("160100")), ("5,000 (원)", Decimal("5000")), ("4,054.5", Decimal("4054.5")),
    ("12,000원", Decimal("12000")), ("12,000 (주1)", Decimal("12000")), ("160100", Decimal("160100")),
    # formats seen in real filings
    ("취득(3,975원)", Decimal("3975")), ("(165,000)", Decimal("165000")), ("9,288 (-)", Decimal("9288")),
    ("11,400(   )", Decimal("11400")), ("-/24,000", Decimal("24000")),
    ("-", None), ("0", None), ("(주1)", None), ("10,000~12,000", None), ("1,0000", None),
])
def test_to_price(raw, expected):
    assert filings.to_price(raw) == expected


@pytest.mark.parametrize("unit, text, expected", [
    ("20261019", "2026.10.19", "2026-10-19"),
    (None, "2026년 07월 08일", "2026-07-08"),
    ("-", "2026.1.5", "2026-01-05"),
    ("-", "-", None),
])
def test_to_date(unit, text, expected):
    assert filings.to_date(unit, text) == expected


def test_read_document_zip_roundtrip_and_limits():
    xml = make_document([trade_row("장내매수(+)", "20261019", "10", "1,000")])
    assert filings.read_document_zip(zip_document(xml)) == xml

    with pytest.raises(filings.DocumentError):
        filings.read_document_zip(b'{"status":"014","message":"no file"}')
    with pytest.raises(filings.DocumentError):
        filings.read_document_zip(b"PK" + b"\0" * (filings.MAX_ZIP_BYTES + 1))

    corrupted = bytearray(zip_document(xml))
    start = corrupted.index(b"<?xml")
    corrupted[start:start + 5] = b"<!xml"  # breaks the CRC check
    with pytest.raises(filings.DocumentError):
        filings.read_document_zip(bytes(corrupted))


def test_list_item_normalizes_market_and_amendments():
    item = filings.list_item({
        "corp_code": "00123456", "corp_name": "테스트", "stock_code": "123456 ", "corp_cls": "N",
        "report_nm": "[기재정정]임원ㆍ주요주주특정증권등소유상황보고서", "rcept_no": "20261021000123",
        "flr_nm": "홍길동", "rcept_dt": "20261021", "rm": "",
    })
    assert item["market"] == "코넥스"
    assert item["stock_code"] == "123456"
    assert item["report_date"] == "2026-10-21"
    assert item["amended"] is True


def test_public_record_is_json_friendly():
    item = filings.list_item({"rcept_no": "20261021000123", "rcept_dt": "20261021", "flr_nm": "", "corp_cls": "Y"})
    doc = filings.parse_document(make_document([trade_row("장내매수(+)", "20261019", "3", "4,054.5")]))
    record = filings.public_record(item, doc)
    assert record["reporter"] == "홍길동"  # falls back to the name in the document
    assert record["trades"][0]["price"] == 4054.5
    assert isinstance(record["buy"]["amount"], int)
