"""Busan Port cruise forecasts | ETS-only Streamlit dashboard.

Run (from this folder): python -m streamlit run app.py
The app starts with precomputed, aggregate-only reference results; uploaded CSVs
recalculate every model, validation metric, and 2027 schedule comparison.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    ARRIVAL_START, CUTOFF, KNOWN_ZERO, PASSENGER_START,
    arrival_monthly, compare_schedule, forecast_arrivals, forecast_passengers,
    load_arrival_rows, load_schedule, missing_passenger_months, passenger_monthly,
)
from validation import validate_2026

HERE = Path(__file__).resolve().parent
st.set_page_config(
    page_title="부산항 크루즈 예측 분석 대시보드",
    page_icon="⚓", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family:'Noto Sans KR','Malgun Gothic',sans-serif;
}
[data-testid="stAppViewContainer"] {background:#f6f8fe;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1520px; padding:2rem 2.8rem 4rem;}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#1b2e53 0%,#101e3a 90%);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] small {
   color:#ecf3ff;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#bdcdec;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stExpander"] {border-color:#42608a;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section button,
[data-testid="stSidebar"] button {font-weight:750;}
[data-testid="stSidebar"] [role="radiogroup"] label {
   border:1px solid #526988; padding:11px 9px; border-radius:9px;
   margin:3px 0; background:rgba(255,255,255,.07);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
   background:#d7e8ff; border-color:#9bc4ff;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span {
   color:#112c5b!important; font-weight:850;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-weight:650;}
[data-testid="stSidebar"] .stCheckbox label {color:#e1efff;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
   color:white; background:#2075e6; border:0;
}
h1,h2,h3 {color:#152c55; letter-spacing:-.045em;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
   gap:8px; border-bottom:1px solid #d9e2f1;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
   min-width:150px; border-radius:8px 8px 0 0; background:#e9effa;
   font-weight:850; color:#24395d; height:46px; padding:0 21px;
}
[data-testid="stTabs"] [aria-selected="true"] {
   background:#1b70d7!important; color:#fff!important;
}
[data-testid="stDataFrame"] {border:1px solid #d6e4f6; border-radius:11px;}
.nav-brand {padding:10px 4px 13px; text-align:center;}
.nav-brand .ship {font-size:39px;line-height:1.12;color:white;}
.nav-brand .brand {font-size:18px;font-weight:900;line-height:1.53;color:#fff;}
.nav-brand .tiny {font-size:10px;letter-spacing:.11em;color:#aabedd;margin-top:7px;}
.top-chip {display:inline-block;background:#253550;color:#fff;font-size:13px;
   font-weight:750;padding:7px 14px;border-radius:8px;margin:0 0 9px;}
.main-heading {font-size:35px;font-weight:900;letter-spacing:-.065em;line-height:1.35;color:#132a51;}
.main-description {font-size:13px;color:#5e708a;margin:5px 0 13px;}
.model-name {font-weight:900;font-size:18px;color:#18335e;margin:17px 0 4px;}
.model-details {font-size:13px;color:#586c89;line-height:1.8;margin-bottom:16px;}
.small-section-title {font-weight:850;font-size:16px;color:#18345e;margin:26px 0 12px;}
.info-tile {min-height:125px;background:linear-gradient(135deg,#ebf2ff 0%,#e3edff 100%);
    padding:20px 16px;border-radius:13px;border:1px solid #e1eaff;box-shadow:0 2px 6px #22365b08;}
.info-tile .tile-label {color:#1b4c9c;font-size:13px;font-weight:900;}
.info-tile .tile-value {margin-top:11px;font-weight:900;color:#152b50;font-size:17px;}
.info-tile .tile-sub {color:#5d7193;font-size:11px;margin-top:7px;line-height:1.65;}
.metric-tile {min-height:111px;background:linear-gradient(135deg,#eaf1fe,#e4eeff);padding:18px 19px;
 border:1px solid #e2eafa;border-radius:13px;box-shadow:0 2px 7px #163d7907;}
.metric-tile .metric-heading {font-weight:800;color:#36547a;font-size:12px;}
.metric-tile .metric-value {font-weight:900;color:#132c57;font-size:26px;letter-spacing:-.05em;margin-top:9px;}
.metric-tile .metric-value.small {font-size:21px;}
.metrics-table {border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;
 border:1px solid #d7e2f2;text-align:center;font-size:12px;}
.metrics-table th {padding:14px 7px;background:#eef2f9;color:#263a5e;font-weight:850;border:1px solid #d7e2f2;}
.metrics-table td {padding:17px 6px;background:#e1ecff;color:#14356e;font-weight:850;border:1px solid #d7e2f2;}
.reason-box {background:linear-gradient(135deg,#ecf2ff,#e2ecff);border:1px solid #dbe6ff;
 padding:17px 18px;border-radius:12px;font-size:13px;color:#314b72;line-height:1.8;min-height:140px;}
.reason-box strong {font-weight:900;color:#1a4383;font-size:14px;}
.reason-box ul {padding-left:18px;margin:6px 0 0;}
.result-box {background:#f3f6ff;border:1px solid #d9e6fc;border-radius:13px;
 padding:15px 17px;font-size:12px;line-height:1.8;color:#344b70;}
.result-box .head {font-size:14px;font-weight:900;color:#1f4081;margin-bottom:10px;}
.result-box .result-item {padding:7px 0;border-bottom:1px solid #dee7f5;}
.result-box .result-item:last-child {border-bottom:0;}
.dot-check {color:#1878df;font-weight:900;margin-right:6px;}
.foot-note {color:#667998;font-size:12px;line-height:1.85;margin:11px 0;}
.status-line {background:#eef4ff;border:1px solid #dae7f9;border-radius:10px;
 padding:10px 13px;color:#344d75;font-size:12px;margin:5px 0 15px;}
@media(max-width:850px){.block-container{padding:1.3rem 1rem 3rem;}
 .main-heading{font-size:27px;}.metric-tile .metric-value{font-size:20px;}}
</style>
""", unsafe_allow_html=True)


def format_value(value: float, unit: str, digits: int = 0) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{value:,.{digits}f}{unit}"


def safe_float(series: pd.Series) -> float:
    if series.isna().all():
        return np.nan
    return float(series.sum(min_count=1))


def render_info(label: str, value: str, sub: str) -> None:
    st.markdown(f'''<div class="info-tile"><div class="tile-label">{label}</div>
     <div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>''', unsafe_allow_html=True)


def render_metric(label: str, value: str, compact: bool = False) -> None:
    smaller = " small" if compact else ""
    st.markdown(f'''<div class="metric-tile"><div class="metric-heading">{label}</div>
      <div class="metric-value{smaller}">{value}</div></div>''', unsafe_allow_html=True)


def load_preview() -> dict:
    folder = HERE / "preview"
    def frame(filename: str) -> pd.DataFrame:
        return pd.read_csv(folder / filename, encoding="utf-8-sig")
    m = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return {
        "source": "preview", "source_info": m,
        "arrivals": frame("arrivals_2027.csv"),
        "passengers": frame("passengers_2027.csv"),
        "arrival_comparison": frame("arrival_compare_2027.csv"),
        "passenger_comparison": frame("passenger_compare_2027.csv"),
        "backtests": {
            "arrivals": {"metrics": m["arrivals_metrics"], "rows": frame("arrivals_backtest_2026.csv")},
            "passengers": {"metrics": m["passengers_metrics"], "rows": frame("passengers_backtest_2026.csv")},
        },
        "arrivals_meta": m["arrivals_meta"],
        "passengers_meta": m["passengers_meta"],
        "compare_info": m["comparison_info"],
        "missing_arrival": [], "impute_log": pd.DataFrame(),
        "input_info": {"ignored_after_cutoff": 20, "raw_arrival_rows": m["source_arrivals"],
                       "duplicate_count": 0},
    }


@st.cache_data(show_spinner=False, max_entries=4)
def calculate(raw: bytes, schedule: bytes | None,
              zero_months: tuple[str, ...], remove_duplicates: bool,
              confirmed_schedule_complete: bool) -> dict:
    rows, info = load_arrival_rows(raw, remove_duplicates)
    absent = missing_passenger_months(rows)
    unknown = sorted(set(absent) - set(zero_months))
    if unknown:
        raise ValueError("승객 자료의 미확인 무입항 월: " + ", ".join(unknown))
    amonth, no_arrival_months = arrival_monthly(rows)
    pmonth, log = passenger_monthly(rows, zero_months)
    arrival, arrival_meta = forecast_arrivals(amonth)
    passenger, passenger_meta = forecast_passengers(pmonth)
    sch = load_schedule(schedule) if schedule is not None else None
    pcompare, acompare, compare_info = compare_schedule(
        passenger, arrival, sch, complete_schedule=confirmed_schedule_complete
    )
    validation = validate_2026(amonth, pmonth)
    return {
        "source": "uploaded", "source_info": {}, "arrivals": arrival,
        "passengers": passenger, "arrival_comparison": acompare,
        "passenger_comparison": pcompare, "backtests": validation,
        "arrivals_meta": arrival_meta, "passengers_meta": passenger_meta,
        "compare_info": compare_info, "missing_arrival": no_arrival_months,
        "impute_log": log, "input_info": info,
    }


def forecast_chart(forecast: pd.DataFrame, compare: pd.DataFrame,
                   model: str) -> go.Figure:
    is_passenger = model == "passengers"
    point = "예측승객수" if is_passenger else "예측입항수"
    plan = "예정 승객수" if is_passenger else "예정입항건수합계"
    period = forecast["연월"].tolist()
    names = [f"{int(x[-2:])}월" for x in period]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=names, y=forecast["95% 상한"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=names, y=forecast["95% 하한"], mode="lines",
        fill="tonexty", fillcolor="rgba(109,151,238,.20)",
        line=dict(width=0), name="95% 근사 예측구간",
        hovertemplate="%{x}<br>구간 하한 %{y:,.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=names, y=compare[plan], name="2027년 예정 승객수" if is_passenger else "2027년 예정 입항수",
        marker=dict(color="#a6c8fa"), opacity=.75,
        hovertemplate="%{x}<br>예정 %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=names, y=forecast[point], name="ETS 예측", mode="lines+markers",
        line=dict(color="#1569d8", width=3.3), marker=dict(size=7, color="#1569d8"),
        hovertemplate="%{x}<br>예측 %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=410, barmode="overlay", hovermode="x unified",
        margin=dict(l=7, r=5, t=12, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family="Noto Sans KR, Malgun Gothic, sans-serif", size=11, color="#35516f"),
        legend=dict(orientation="h", y=1.18, yanchor="top", x=0, font=dict(size=11)),
        yaxis=dict(title="승객수 (명)" if is_passenger else "입항수 (건)",
                   rangemode="tozero", zeroline=False, gridcolor="#ebf0f7", tickformat=",.0f"),
        xaxis=dict(title=None, categoryorder="array", categoryarray=names,
                   tickmode="array", tickvals=names, showgrid=False),
    )
    return fig


def performance_table(scores: dict, unit: str) -> None:
    def num(k: str) -> str:
        return f"{scores[k]:,.2f}" if np.isfinite(scores[k]) else "—"
    st.markdown(f'''
    <table class="metrics-table">
      <thead><tr><th>구분</th><th>MAE ({unit})</th><th>RMSE ({unit})</th><th>MAPE</th><th>WAPE</th></tr></thead>
      <tbody><tr><td>ETS (선정)</td><td>{num('MAE')}</td><td>{num('RMSE')}</td>
      <td>{num('MAPE')}%</td><td>{num('WAPE')}%</td></tr></tbody>
    </table>''', unsafe_allow_html=True)


def section_intro(model: str, result: dict) -> None:
    is_p = model == "passengers"
    if is_p:
        st.markdown('<div class="model-name">예측 모델: ETS (감쇠 추세 계절모형)</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-details">승객 수의 수준·추세·계절성을 반영하고, 감쇠 계수(φ)를 이용해 장기 예측 시 추세가 지나치게 확대되는 것을 완화하는 지수평활 모형입니다.</div>', unsafe_allow_html=True)
        model_form, period, unit = "ETS (A, Ad, A)", "2023.01 ~ 2026.09", "명"
        explanation = "승객 수는 최근 기간의 연속된 기록을 바탕으로 학습하며, 확인된 무입항 월과 일부 결측 보완을 구분해 반영합니다."
        reasons = ["승객 수의 추세와 월별 계절성을 동시에 설명할 수 있습니다.", "감쇠 추세로 장기 예측의 추세 확대를 제한합니다."]
    else:
        st.markdown('<div class="model-name">예측 모델: ETS (가법 계절모형)</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-details">월별 입항건수의 수준과 계절 패턴을 지수평활로 추정하는 모형입니다. 추세 항 없이 계절성을 반영합니다.</div>', unsafe_allow_html=True)
        model_form, period, unit = "ETS (A, N, A)", "2013.01 ~ 2026.09", "건"
        explanation = "입항수는 장기 입항 기록을 사용해 시계열의 연속성을 유지하고, 매월 입항 건수의 계절 변화를 반영합니다."
        reasons = ["뚜렷한 월별 계절 패턴을 간결한 구조로 반영합니다.", "모형 구조가 단순하여 운영계획 검토 시 해석하기 쉽습니다."]
    a,b,c = st.columns(3, gap="small")
    with a: render_info("⚙️  모델 형태", model_form, "가법 오차 · 감쇠 추세 · 가법 계절" if is_p else "가법 오차 · 추세 없음 · 가법 계절")
    with b: render_info("▤  학습 기간", period, "45개월 · 최근 승객 기록" if is_p else "165개월 · 장기 입항 기록")
    with c: render_info("▥  예측 기간", "2026.10 ~ 2027.12", "15개월 연속 예측 → 2027년만 표시")
    st.markdown(f'<div class="foot-note">{explanation}</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-section-title">모델 성능 (2026.01 ~ 2026.09 실제값 검증)</div>', unsafe_allow_html=True)
    tleft, tright = st.columns([1.65, 1], gap="medium")
    with tleft:
        scores = result["backtests"][model]["metrics"]
        performance_table(scores, unit)
        st.caption("2025년 12월까지로 별도 학습하고 2026년 1~9월을 9개월 선행 예측한 결과입니다. 2027년 예측 성능을 보장하지 않습니다.")
    with tright:
        reason_html = "".join(f"<li>{x}</li>" for x in reasons)
        st.markdown(f'<div class="reason-box"><strong>🔵 선정 이유</strong><ul>{reason_html}</ul></div>', unsafe_allow_html=True)
    with st.expander("지표 정의 및 검증 범위 확인"):
        st.markdown("**오차** = 실제 − 예측. **MAE** = 절대오차 평균, **RMSE** = 제곱오차 평균의 제곱근, **MAPE** = 실제가 0이 아니고 유효한 월의 절대백분율오차 평균, **WAPE** = 유효월 절대오차 합계 ÷ 실제 합계 × 100.")
        st.caption(f"검증에 사용한 실제 유효월: {scores['months']}개월 / MAPE 평가월: {scores['mape_months']}개월. 95% 구간 포함률은 별도로 실제값을 구간과 대조한 지표이며, 예정 스케줄은 여기에 사용하지 않습니다.")
        st.caption("학습 시기: 2025.12까지 / 검증: 2026.01~09. 2027 전망 생성 시에는 별도로 2026.09까지 재학습합니다.")


def show_report(model: str, result: dict) -> None:
    is_p = model == "passengers"
    prediction = result[model]
    compare = result["passenger_comparison"] if is_p else result["arrival_comparison"]
    pred_col = "예측승객수" if is_p else "예측입항수"
    plan_col = "예정 승객수" if is_p else "예정입항건수합계"
    units = "명" if is_p else "건"
    predicted_total = float(prediction[pred_col].sum())
    planned_total = safe_float(compare[plan_col])
    valid = compare.loc[compare[plan_col].notna() & (compare[plan_col] >= 0)].copy()
    valid_pos = valid[valid[plan_col] > 0].copy()
    is_complete = len(valid) == 12
    number_digits = 0 if is_p else 2
    d = predicted_total - planned_total
    percent = (d / planned_total * 100) if np.isfinite(planned_total) and planned_total > 0 else np.nan
    metric_diff = "—" if not np.isfinite(d) else f"{d:+,.{0 if is_p else 2}f}{units} ({percent:+.2f}%)" if np.isfinite(percent) else f"{d:+,.2f}{units}"
    m1,m2,m3 = st.columns(3, gap="small")
    with m1: render_metric(f"2027년 예측 {'승객수' if is_p else '입항수'}", format_value(predicted_total, units, number_digits))
    with m2: render_metric("2027년 예정 승객수" if is_p else "2027년 입항 예정건수", format_value(planned_total, units))
    with m3: render_metric("차이 (예측 − 예정)", metric_diff, compact=True)
    if not is_complete:
        st.warning(f"예정 스케줄이 확인된 달은 {len(valid)}/12개월입니다. 예정 합계와 차이는 업로드된 월만 반영했으므로 연간 정확한 비교로 해석하지 마세요.")
    st.markdown(f'<div class="small-section-title">2027년 월별 {"승객수" if is_p else "입항수"} 예측 및 예정 스케줄 비교</div>', unsafe_allow_html=True)
    graphic, summary = st.columns([3.45, 1.17], gap="small")
    with graphic:
        st.plotly_chart(forecast_chart(prediction, compare, model), use_container_width=True, key=f"{model}_2027_plot")
    with summary:
        if len(valid_pos):
            ape = ((valid_pos[pred_col] - valid_pos[plan_col]).abs() / valid_pos[plan_col] * 100).mean()
            larger = valid.loc[valid[pred_col] > valid[plan_col], "연월"].str[-2:].astype(int).tolist()
            above = ", ".join(f"{m}월" for m in larger) if larger else "없음"
            aligned = valid.merge(prediction[["연월", "95% 하한", "95% 상한"]], on="연월", validate="one_to_one")
            inrange = int(aligned[plan_col].between(aligned["95% 하한"], aligned["95% 상한"]).sum())
            items = [
                f"월별 예정 대비 절대차이율 평균: <b>{ape:.2f}%</b>",
                f"예측이 예정값보다 큰 월: <b>{above}</b>",
                f"예정값이 예측구간 내에 있는 월: <b>{inrange}/{len(aligned)}개월</b>",
            ]
        else:
            items = ["스케줄을 업로드하면 월별 차이율을 비교할 수 있습니다."]
        m = result["passengers_meta" if is_p else "arrivals_meta"]
        annual = m.get("annual_interval", [np.nan, np.nan])
        items.append(f"연간 95% 근사 구간: <b>{format_value(annual[0], units, 0 if is_p else 1)} ~ {format_value(annual[1], units, 0 if is_p else 1)}</b>")
        rows_html = "".join(f'<div class="result-item"><span class="dot-check">✓</span>{item}</div>' for item in items)
        st.markdown(f'<div class="result-box"><div class="head">주요 결과</div>{rows_html}</div>', unsafe_allow_html=True)
    st.caption("음영은 95% 근사 예측구간입니다. 예정 스케줄을 실제 관측값으로 간주하지 않으며, 예정값의 구간 내 위치는 실제 포함률이 아닙니다.")
    if is_p:
        st.markdown('<div class="foot-note">※ <b>예정 승객수</b>는 2027년 스케줄의 <b>승객정원수</b>를 월별로 합산한 참고값입니다. 실제 탑승 승객수가 아니며 선박의 실제 정원수가 검증·확정되었다는 의미도 아닙니다.</div>', unsafe_allow_html=True)
        st.caption("승객 표의 차이 = 예정 승객수 − 예측승객수 (원본 승객 분석의 부호 기준). 위 KPI는 화면 비교를 위해 예측 − 예정으로 표시합니다.")
        shown = compare.rename(columns={"예측승객수": "예측 승객수", "예정대비절대차이율(%)": "예정 대비 절대차이율(%)", "차이": "차이(예정−예측)"})
    else:
        st.caption("입항 표의 차이 = 예측입항수 − 예정입항건수합계 (원본 다이렉트 노트북과 동일한 부호 기준).")
        shown = compare.drop(columns=["실제 예정 입항수"], errors="ignore").rename(columns={
            "예측입항수": "예측 입항수", "예정입항건수합계": "입항 예정건수",
            "예정건수대비절대차이율(%)": "예정 대비 절대차이율(%)", "차이": "차이(예측−예정)",
        })
    st.markdown('<div class="small-section-title">2027년 월별 비교표 (1월~12월)</div>', unsafe_allow_html=True)
    formatted = shown.copy()
    for key in formatted.select_dtypes(include=["number"]).columns:
        if key.endswith("(%)"):
            formatted[key] = formatted[key].map(lambda n: "—" if pd.isna(n) else f"{n:.2f}%")
        elif key.startswith("예측 ") or key.startswith("차이("):
            formatted[key] = formatted[key].map(lambda n: "—" if pd.isna(n) else f"{n:,.2f}")
        else:
            formatted[key] = formatted[key].map(lambda n: "—" if pd.isna(n) else f"{n:,.0f}")
    st.dataframe(formatted, hide_index=True, use_container_width=True, height=455)
    st.download_button(
        "⬇ 월별 비교표 CSV 내려받기", data=shown.to_csv(index=False).encode("utf-8-sig"),
        file_name="2027_passengers_ets.csv" if is_p else "2027_arrivals_ets.csv",
        mime="text/csv", key=f"download_{model}",
    )
    with st.expander("예측 방법과 지표 정의"):
        st.write(f"2026년 9월까지 학습한 **{'ETS(A,Ad,A)' if is_p else 'ETS(A,N,A)'}** 모형으로 **2026.10~2027.12를 한 번에 15개월 예측**하고, 이 중 2027년 12개월만 표시합니다.")
        if not is_p:
            st.write("입항수는 **다이렉트 예측만** 사용합니다. 2026년 10~12월 예측치를 학습에 추가하여 재적합하지 않습니다. 근사 구간은 잔차를 복원추출한 미래 경로 기반입니다.")
        else:
            st.write("승객수는 **감쇠 추세 ETS** 모형을 적용하고 1만 개 미래 경로를 이용해 95% 근사 예측구간을 계산합니다. 학습 중 일부 결측은 과거 동일 선박의 기록으로만 보완하며 실제 승객 합계와 구분합니다.")
        st.write("**오차 = 실제 − 예측. MAPE = 실제값 0 또는 결측을 제외한 월별 절대백분율오차 평균. WAPE = 절대오차 합계 ÷ 실제 합계 × 100. 포함률 = 실제값이 95% 구간 안에 든 유효월의 비율.**")
        st.caption("2027년 실제 실적이 아직 없어 2027년 MAPE·WAPE·포함률은 산출할 수 없습니다. 예정 스케줄 대비 차이율은 예측 정확도가 아닙니다. 95% 구간 역시 실제 포함률 95%를 보장하지 않습니다.")


with st.sidebar:
    st.markdown('''<div class="nav-brand"><div class="ship">⚓</div>
       <div class="brand">부산항 크루즈<br>예측 분석 대시보드</div>
       <div class="tiny">BUSAN PORT · ETS FORECAST</div></div>''', unsafe_allow_html=True)
    choice = st.radio("메뉴", ["👤  승객수 예측", "⚓  입항건수 예측"], label_visibility="collapsed")
    st.divider()
    st.markdown("**📁 데이터 업로드**")
    raw_upload = st.file_uploader("입출항 원자료 CSV", type="csv", key="uploaded_raw",
                                  help="i.csv · 입출/입항일시/호출부호/선명/승객 열")
    schedule_upload = st.file_uploader("2027년 예정 스케줄 CSV", type="csv", key="uploaded_sched",
                                       help="입항예정일/선명/승객정원수 열")
    st.caption("업로드하지 않으면 제공된 데이터로 미리 계산해 둔 집계 결과를 보여줍니다. ZIP에 원본 CSV는 포함하지 않았습니다.")
    remove_dups = st.checkbox("완전 중복 행 확인 후 제거", value=False)
    complete_2026 = st.checkbox("2026.09까지 실제 자료 수집 완료 확인", value=False)
    schedule_complete = st.checkbox("2027년 예정 스케줄 12개월 확인", value=False)
    if raw_upload is not None:
        try:
            upload_rows, upload_info = load_arrival_rows(raw_upload.getvalue(), remove_dups)
            missing = missing_passenger_months(upload_rows)
            chosen_zeros = st.multiselect("무입항으로 확인된 월만 선택", missing,
                 default=[z for z in missing if z in KNOWN_ZERO],
                 help="승객 수 무기록 월을 자동으로 0명으로 간주하지 않습니다.")
            st.caption(f"입항 행: {upload_info['raw_arrival_rows']:,}건 · 학습 제외(2026.09 이후): {upload_info['ignored_after_cutoff']:,}건")
        except Exception as err:
            st.error(f"원자료를 읽을 수 없습니다: {err}")
            chosen_zeros = []
    else:
        chosen_zeros = []
    run = st.button("▶ 예측 결과 업데이트", type="primary", use_container_width=True,
                    disabled=raw_upload is None or not complete_2026)
    with st.expander("CSV 입력 양식"):
        for fn, text in [("raw_template.csv", "원자료 양식"), ("schedule_template.csv", "2027년 스케줄 양식")]:
            st.download_button(text, (HERE / "templates" / fn).read_bytes(),
                               file_name=fn, mime="text/csv", key=f"tmp_{fn}")

preview = load_preview()
if raw_upload is not None:
    this_hash = hashlib.sha256(raw_upload.getvalue()
        + (schedule_upload.getvalue() if schedule_upload is not None else b"")
        + str((tuple(sorted(chosen_zeros)), remove_dups, schedule_complete)).encode()).hexdigest()
else:
    this_hash = None

if run:
    try:
        with st.spinner("입항수·승객수 ETS 예측, 2026년 검증, 2027년 비교를 계산하고 있습니다..."):
            updated = calculate(raw_upload.getvalue(),
                schedule_upload.getvalue() if schedule_upload is not None else None,
                tuple(sorted(chosen_zeros)), remove_dups, schedule_complete)
        st.session_state["calculated"] = updated
        st.session_state["calculate_hash"] = this_hash
        st.toast("2027년 예측 결과를 업데이트했습니다.", icon="✅")
    except Exception as err:
        st.error(f"새 자료 계산 실패: {err}")

is_updated = (this_hash is not None and st.session_state.get("calculate_hash") == this_hash)
result = st.session_state.get("calculated", preview) if is_updated else preview

selected = "passengers" if choice.startswith("👤") else "arrivals"
page_number = "1-1" if selected == "passengers" else "1-2"
title = "승객수 예측" if selected == "passengers" else "입항건수 예측"
subtitle = "과거 입항 데이터를 기반으로 크루즈 승객 수를 예측합니다." if selected == "passengers" else "과거 입항 기록을 기반으로 크루즈 월별 입항건수를 예측합니다."
st.markdown(f'<span class="top-chip">{page_number}. {title} · ETS 모델</span>', unsafe_allow_html=True)
st.markdown(f'<div class="main-heading">{title}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-description">{subtitle}</div>', unsafe_allow_html=True)
if is_updated:
    st.markdown('<div class="status-line">✅ <b>업로드한 원자료로 재계산한 결과</b> · 2026.09 학습 마감 · 2027.01~12 표시</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-line">ℹ️ <b>기본 미리보기</b> · 제공받은 원자료로 미리 계산한 집계 결과입니다. 새 CSV를 업로드한 경우 왼쪽의 업데이트 버튼을 눌러 결과에 반영하세요.</div>', unsafe_allow_html=True)

intro_tab, forecast_tab = st.tabs(["모델 소개", "2027년 예측"])
with intro_tab:
    section_intro(selected, result)
with forecast_tab:
    show_report(selected, result)

st.divider()
with st.expander("🎬 원자료 → ETS 모델 → 2027년 결과 생성 과정 (14초 애니메이션)"):
    vid = HERE / "assets" / "workflow_demo.mp4"
    if vid.exists():
        st.video(str(vid))
        st.caption("설명용 애니메이션입니다. 영상에 나타나는 숫자·그래프는 실제 모델 출력과 무관합니다.")
    else:
        st.warning("assets/workflow_demo.mp4가 없습니다. 앱 폴더 전체를 배포해주세요.")
with st.expander("데이터 품질·재현성 안내"):
    st.markdown("**입항수:** 2013.01~2026.09(165개월), ETS(A,N,A), 15개월 연속 다이렉트 예측. 무기록 월은 입항 노트북의 기준대로 0건으로 집계합니다.")
    st.markdown("**승객수:** 2023.01~2026.09(45개월), ETS(A,Ad,A). 무입항으로 확인한 월만 0명으로 처리하고, 승객 결측은 학습용 자료에 한해서 과거 동일 호출부호 중앙값으로 보완합니다.")
    st.markdown("**근사 예측구간:** 입항수는 잔차 부트스트랩 1,000경로, 승객수는 10,000경로 기반입니다. 연간 95% 구간은 경로별 연간 합계에서 산출하며 월별 구간을 단순 합하지 않습니다.")
    st.caption("이 프로그램은 입력 CSV를 외부 API로 전송하도록 설계하지 않았습니다. Streamlit Cloud에 올리는 경우 서버 및 접근 정책을 확인하고 민감한 원본 CSV를 공개 저장소에 올리지 마세요.")
    st.caption("동일한 419.19건을 재현하려면 requirements.txt에서 지정한 statsmodels·NumPy·SciPy 환경을 사용하는 것이 권장됩니다. 모수 추정 환경에 따라 미세 차이가 발생할 수 있습니다.")
    if is_updated:
        n = result["input_info"]["ignored_after_cutoff"]
        st.write(f"원자료의 2026.09 이후 학습 제외 입항 기록: **{n}건**")
        if not result["impute_log"].empty:
            st.markdown("**학습용 승객 결측 보완 기록**")
            st.dataframe(result["impute_log"], hide_index=True, use_container_width=True)

