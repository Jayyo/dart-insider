import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Markup below mirrors real 임원ㆍ주요주주특정증권등소유상황보고서 documents,
# trimmed to the cells the parser reads. Names and numbers are made up.
HEADER = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="dart4.xsd">
<DOCUMENT-NAME ACODE="00634">임원ㆍ주요주주 특정증권등 소유상황보고서</DOCUMENT-NAME>
<TABLE-GROUP ACLASS="IFR_NM" ADELETETABLE="N"><TABLE ACLASS="EXTRACTION" AFIXTABLE="Y" WIDTH="745" BORDER="1"><TBODY>
<TR ACOPY="Y" ADELETE="Y" HEIGHT="30">
<TD ALIGN="CENTER">보고구분</TD><TU ALIGN="CENTER" AUNIT="RPT_DST" AUNITVALUE="2">변동</TU>
<TD ALIGN="CENTER">보고자 구분</TD><TU COLSPAN="2" AUNIT="IFR_TP" AUNITVALUE="{ifr_tp_code}">{ifr_tp}</TU>
</TR>
<TR><TD>한     글</TD><TE ALIGN="CENTER" ACODE="IFR_NM" WIDTH="128">{name}</TE></TR>
<TR><TD>임원(등기여부)</TD><TU ALIGN="CENTER" AUNIT="STF_RYN" AUNITVALUE="{ryn_code}">{ryn}</TU>
<TD>직위명</TD><TE ALIGN="CENTER" ACODE="STF_PSM">{position}</TE></TR>
<TR><TD>주요주주</TD><TU COLSPAN="3" AUNIT="MAIN_SH" AUNITVALUE="{main_code}">{main}</TU></TR>
</TBODY></TABLE></TABLE-GROUP>
<TABLE-GROUP ACLASS="PS_CPT" ADELETETABLE="N"><TABLE><TBODY>
<TR><TD>직전 보고서</TD><TE ACODE="BFR_PS_CPT_CNT">10,000</TE><TE ACODE="BFR_PS_CPT_RT">1.25</TE></TR>
<TR><TD>이번 보고서</TD><TE ACODE="AFR_PS_CPT_CNT">12,000</TE><TE ACODE="AFR_PS_CPT_RT">1.50</TE></TR>
</TBODY></TABLE></TABLE-GROUP>
"""

TRADE_TABLE = """<TABLE-GROUP ACLASS="RPT_RSN" ADELETETABLE="N">
<TABLE ACLASS="EXTRACTION" AFIXTABLE="N" WIDTH="822" BORDER="1">
<THEAD><TR><TH>보고사유</TH><TH>변동일*</TH><TH>특정증권등의종류</TH><TH>변동전</TH><TH>증감</TH><TH>변동후</TH><TH>취득/처분단가(원)**</TH></TR></THEAD>
<TBODY>
{rows}
<TR ACOPY="N" ADELETE="N" HEIGHT="30">
<TD COLSPAN="3" ALIGN="CENTER">합  계</TD>
<TE ALIGN="RIGHT" ACODE="BFR_STK_SUM">10,000</TE><TE ALIGN="RIGHT" ACODE="MDF_STK_SUM">2,000</TE>
<TE ALIGN="RIGHT" ACODE="AFR_STK_SUM">12,000</TE><TE ALIGN="RIGHT" ACODE="ACI_AMT_SUM2">-</TE>
</TR>
</TBODY></TABLE></TABLE-GROUP>
</DOCUMENT>
"""

ROW = """<TR ACOPY="Y" ADELETE="N" HEIGHT="53">
<TU ALIGN="CENTER" AUNIT="RPT_RSN" AUNITVALUE="{code}">{reason}</TU>
<TU ALIGN="CENTER" AUNIT="MDF_DM" AUNITVALUE="{ymd}">{date_text}</TU>
<TU ALIGN="CENTER" AUNIT="STR_KND" AUNITVALUE="1">{security}</TU>
<TE ALIGN="RIGHT" ACODE="BFR_STK_CNT">{before}</TE>
<TE ALIGN="RIGHT" ACODE="MDF_STK_CNT">{change}</TE>
<TE ALIGN="RIGHT" ACODE="AFR_STK_CNT">{after}</TE>
<TE ALIGN="RIGHT" ACODE="ACI_AMT2">{price}</TE>
<TE ALIGN="CENTER" ACODE="RMK">-</TE>
<TU ALIGN="CENTER" AUNIT="TRD_PLN_RD" AUNITVALUE="{plan}">{plan_text}</TU>
</TR>"""


def trade_row(reason, ymd, change, price, security="보통주", code="01", before="10,000", after="12,000",
              date_text=None, plan="-"):
    return ROW.format(
        code=code, reason=reason, ymd=ymd, date_text=date_text or f"{ymd[:4]}년 {ymd[4:6]}월 {ymd[6:]}일",
        security=security, before=before, change=change, after=after, price=price,
        plan=plan, plan_text=plan if plan == "-" else f"{plan[:4]}.{plan[4:6]}.{plan[6:]}",
    )


def make_document(rows, name="홍 길 동", ryn=("Y", "등기임원"), position="대표이사", main=("4", "-"),
                  ifr_tp=("D", "개인(국내)")):
    head = HEADER.format(name=name, ryn_code=ryn[0], ryn=ryn[1], position=position, main_code=main[0],
                         main=main[1], ifr_tp_code=ifr_tp[0], ifr_tp=ifr_tp[1])
    return head + TRADE_TABLE.format(rows="\n".join(rows))


def zip_document(xml, name="20261021000001.xml"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, xml.encode("utf-8"))
    return buf.getvalue()
