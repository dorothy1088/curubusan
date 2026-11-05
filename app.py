"""부산항 2027 크루즈 입항/승객 ETS 전망 — Streamlit 웹 앱."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (
    ARRIVAL_START, PASSENGER_START, CUTOFF, KNOWN_ZERO,
    load_arrival_rows, missing_passenger_months, arrival_monthly, passenger_monthly,
    forecast_arrivals, forecast_passengers, load_schedule, compare_schedule,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="부산항 크루즈 | 2027 전망 대시보드",
    page_icon="⚓", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family:'Noto Sans KR',sans-serif;
}
[data-testid="stAppViewContainer"] { background:#f7f9fd; }
[data-testid="stHeader"] { background:#f7f9fd; }
[data-testid="stSidebar"] {background:#10243b; color:#eef5ff; }
[data-testid="stSidebar"] * {color:#e4f0ff !important;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {
 color:#15243a !important;
}
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button {color:#17263a !important; background:#eef6ff; border-radius:9px;}
.block-container {max-width:1510px; padding:1.75rem 2.6rem 4rem;}
h1,h2,h3 {color:#142e47;letter-spacing:-.045em;}
.hero {background:linear-gradient(110deg,#102e4b 0%,#154e68 62%,#168d9b 100%);
  border-radius:22px; padding:32px 38px 29px; color:white; margin-bottom:17px;
  box-shadow:0 13px 35px rgba(8,49,83,.10); position:relative; overflow:hidden;}
.hero .eyebrow {font-size:12px; letter-spacing:.14em; font-weight:800; color:#9be6e3; margin-bottom:10px;}
.hero .headline {font-size:33px;font-weight:800;letter-spacing:-.055em;margin:0 0 10px;line-height:1.42;}
.hero .subtitle {font-size:14px; color:#d3e6ec;line-height:1.8;}
.hero .pill {display:inline-block; margin-top:16px; margin-right:8px; border:1px solid #6197a8;
 background:#ffffff15; color:#e3f7f5;border-radius:30px;padding:6px 12px;font-size:12px;font-weight:700;}
.intro-card {border:1px solid #e2e9f1;background:white;border-radius:15px;padding:18px 21px 16px;min-height:152px;
 box-shadow:0 2px 10px rgba(18,50,82,.03); margin-bottom:4px;}
.intro-card .cap {font-size:11px;letter-spacing:.1em;color:#5f829d;font-weight:800;}
.intro-card .big {font-size:21px;font-weight:800;color:#183b55;margin:8px 0;}
.intro-card .small {font-size:12px;color:#698094;line-height:1.75;}
.stage {border:1px solid #d8e9f0; background:#eef8fa; border-radius:13px; padding:14px 17px; color:#2b5971;line-height:1.8;font-size:13px;}
.tag {display:inline-block;font-size:11px;font-weight:800;padding:4px 8px;border-radius:20px;
  background:#d9f4ef;color:#14686e; margin:1px 3px 4px 0;}
.metric-label {color:#496e89;font-size:13px;}
[data-testid="stMetric"] {border:1px solid #e5eaf2;border-radius:14px;padding:17px 20px;
 background:white;box-shadow:0 2px 8px rgba(20,47,80,.025);}
[data-testid="stMetricValue"] {font-size:26px;font-weight:800;color:#183b55;}
[data-testid="stMetricLabel"] {font-size:12px;color:#6b849b;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:14px;}
[data-testid="stTabs"] [data-baseweb="tab"] {font-weight:700;padding:15px 13px;}
div[data-testid="stVerticalBlock"] h3 {margin-top:18px;}
hr {border-color:#e1e8ef;}
@media (max-width:700px){.block-container{padding:1.3rem 1rem 3rem;} .hero{padding:22px 20px;}
 .hero .headline{font-size:24px;} .intro-card{min-height:unset;} }
</style>
""", unsafe_allow_html=True)


def fmt_number(value: float | None, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.0f}{unit}"


def file_template(name: str) -> bytes:
    return (ROOT / "templates" / name).read_bytes()


@st.cache_data(show_spinner=False)
def parse_original_cached(content: bytes, remove_duplicates: bool):
    return load_arrival_rows(content, remove_duplicates)


@st.cache_data(show_spinner=False)
def forecast_arrivals_cached(monthly: pd.DataFrame):
    return forecast_arrivals(monthly)


@st.cache_data(show_spinner=False)
def forecast_passengers_cached(monthly: pd.DataFrame):
    return forecast_passengers(monthly)


@st.cache_data(show_spinner=False)
def parse_schedule_cached(content: bytes):
    return load_schedule(content)


def plot_forecast(future: pd.DataFrame, observed: pd.DataFrame, value_col: str, unit: str,
                  model: str, plan: pd.Series | None = None, plan_name: str = "2027 예정 스케줄"):
    hist = observed.loc[observed["연월"].between("2025-01", CUTOFF)].copy()
    fig = go.Figure()
    dates = pd.to_datetime(future["연월"])
    fig.add_trace(go.Scatter(
        x=dates, y=future["95% 상한"], mode="lines",
        line=dict(width=0), name="95% 상한", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=future["95% 하한"], mode="lines", fill="tonexty",
        fillcolor="rgba(19,151,163,0.16)", line=dict(width=0),
        name="95% 근사 예측구간", hovertemplate="%{x|%Y-%m}<br>95% 하한: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(hist["연월"]), y=hist[value_col],
        mode="lines+markers", name="과거 실제 실적 (2025~2026.09)",
        line=dict(color="#8f9db1", width=2.5), marker=dict(size=5),
        hovertemplate="%{x|%Y-%m}<br>실제: %{y:,.0f}<extra></extra>",
    ))
    pred_col = [x for x in future.columns if x.startswith("예측")][0]
    fig.add_trace(go.Scatter(
        x=dates, y=future[pred_col], mode="lines+markers", name=f"2027 예측 · {model}",
        line=dict(color="#0b8793", width=3.5), marker=dict(size=7, color="#0b8793"),
        hovertemplate="%{x|%Y-%m}<br>예측: %{y:,.1f}<extra></extra>",
    ))
    if plan is not None and plan.notna().any():
        fig.add_trace(go.Scatter(
            x=dates, y=plan, mode="lines+markers", name=plan_name,
            line=dict(color="#f4a456", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x|%Y-%m}<br>예정: %{y:,.0f}<extra></extra>",
        ))
    fig.add_vrect(x0="2027-01-01", x1="2028-01-01", fillcolor="#eaf8f8", opacity=0.30, line_width=0)
    fig.add_vline(x="2027-01-01", line_width=1.4, line_dash="dot", line_color="#7f9fb3")
    fig.update_layout(
        height=390, margin=dict(l=18, r=18, t=25, b=22), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff", font=dict(family="Noto Sans KR, Malgun Gothic, sans-serif", color="#38536d", size=12),
        hovermode="x unified", legend=dict(orientation="h", y=-0.23, x=0, font=dict(size=11)),
        xaxis=dict(title=None, range=["2025-01-01", "2027-12-23"], tickformat="%Y-%m", dtick="M3",
                   showgrid=True, gridcolor="#edf2f7", linecolor="#d5e1ea"),
        yaxis=dict(title=unit, rangemode="tozero", showgrid=True, gridcolor="#edf2f7", zeroline=False),
    )
    return fig


def download_csv(label: str, frame: pd.DataFrame, filename: str, key: str):
    st.download_button(
        label, data=frame.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=filename, mime="text/csv", key=key,
    )


def show_video():
    video = ROOT / "assets" / "workflow_demo.mp4"
    if video.is_file():
        st.video(str(video))
        st.caption("설명용 애니메이션입니다. 영상의 그래프는 실제 모델 결과가 아니며, 업로드 자료의 수치가 영상에 표시되지는 않습니다.")
    else:
        st.info("설명용 애니메이션이 없는 배포본입니다. assets/workflow_demo.mp4 파일도 함께 배포해주세요.")


st.markdown("""
<div class="hero">
  <div class="eyebrow">BUSAN PORT · CRUISE INTELLIGENCE</div>
  <div class="headline">2027 크루즈 운영 전망 대시보드</div>
  <div class="subtitle">입항 실적과 예정 스케줄을 연결해 2027년 월별 입항·승객 흐름을 한눈에 비교합니다.<br>
  실제 데이터와 미래 예정값을 구분하고, 95% 근사 예측구간으로 전망 범위를 함께 제시합니다.</div>
  <span class="pill">ETS 기반 시계열 예측</span><span class="pill">2027.01 — 2027.12</span>
  <span class="pill">원자료 업로드형</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.0, 1.0, 1.03], gap="medium")
with c1:
    st.markdown('''<div class="intro-card"><div class="cap">ARRIVALS · 입항 건수</div>
<div class="big">ETS(A,N,A)</div><div class="small">2013.01 — 2026.09 · 165개월<br>
장기 입항 패턴과 월별 계절성을 함께 반영</div></div>''', unsafe_allow_html=True)
with c2:
    st.markdown('''<div class="intro-card"><div class="cap">PASSENGERS · 승객 규모</div>
<div class="big">ETS(A,Ad,A)</div><div class="small">2023.01 — 2026.09 · 45개월<br>
최근 승객 기록의 연속성과 자료 완결성을 고려</div></div>''', unsafe_allow_html=True)
with c3:
    st.markdown('''<div class="intro-card"><div class="cap">FORECAST HORIZON · 표시 범위</div>
<div class="big">2027년 12개월</div><div class="small">2026.10 — 2027.12를 연속 예측한 후<br>
2027.01 — 2027.12만 시각화·표에 공개</div></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.header("데이터 설정")
    st.caption("STEP 01 · 데이터 파일 선택")
    raw_upload = st.file_uploader("누적 입출항 원자료 (필수)", type=["csv"], help="입출, 입항일시, 호출부호, 선명, 승객 열이 필요합니다.")
    schedule_upload = st.file_uploader("2027년 입항 예정 스케줄 (선택)", type=["csv"], help="입항예정일, 선명, 승객정원수 열이 필요합니다. 승객정원수는 화면에서 예정 승객수로 표기됩니다.")
    with st.expander("CSV 양식 내려받기"):
        st.download_button("원자료 CSV 헤더", file_template("raw_template.csv"), "raw_template.csv", "text/csv", key="tmpr")
        st.download_button("예정 스케줄 CSV 헤더", file_template("schedule_template.csv"), "schedule_template.csv", "text/csv", key="tmps")
    st.divider()
    st.caption("STEP 02 · 데이터 확인")
    remove_duplicates = st.checkbox("원자료의 완전히 동일한 행 중복 제거", value=False)
    complete_schedule = st.checkbox("2027년 예정 스케줄이 12개월 전체를 포괄함 (없는 월은 0건으로 해석)", value=False)
    confirmed_complete = st.checkbox("2026년 9월까지 실적 수집이 완료되었음을 확인", value=False)
    st.divider()
    st.caption("MODEL REFERENCE")
    st.markdown("**입항수** · ETS(A,N,A)\n\n**승객수** · ETS(A,Ad,A)\n\n**95% 구간** · 확률 경로 시뮬레이션")

original_bytes = raw_upload.getvalue() if raw_upload else None
schedule_bytes = schedule_upload.getvalue() if schedule_upload else None
if original_bytes is None:
    st.markdown("### 업로드부터 전망까지")
    st.markdown("""
    <div class="stage"><span class="tag">01 원자료 입력</span><span class="tag">02 데이터 검증</span>
    <span class="tag">03 ETS 학습·예측</span><span class="tag">04 2027년 비교 분석</span><br>
    왼쪽에서 원자료 CSV를 업로드하면, 입항수·승객수에 맞는 학습자료를 각각 구축합니다.
    입항 예정 스케줄을 추가하면 2027년 월별 예정 승객수·예정 입항 건수와 비교합니다.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 시뮬레이션 흐름 미리보기")
    show_video()
    st.info("현재는 원자료가 없어 실제 모델 수치를 생성하지 않았습니다. CSV를 업로드하고 실적 완료 여부를 확인한 뒤 예측을 실행해주세요.")
    st.stop()

try:
    rows, raw_info = parse_original_cached(original_bytes, remove_duplicates)
except Exception as exc:
    st.error(f"원자료를 읽지 못했습니다: {exc}")
    st.stop()

absent_passenger = missing_passenger_months(rows)
with st.sidebar:
    st.caption("STEP 03 · 무입항 월 확인")
    confirmed_zeros = st.multiselect(
        "승객수 학습기간 중 무입항으로 확인된 월",
        options=absent_passenger,
        default=[m for m in absent_passenger if m in KNOWN_ZERO],
        help="승객수 모델은 무기록 월을 자동 0명 처리하지 않습니다. 실제 무입항이라고 확인된 월만 선택해주세요.",
    )
    st.caption("기존 코드의 확인 월 2023-01, 2023-02, 2023-12, 2024-01을 기본 선택합니다. 나머지는 직접 확인해야 합니다.")

notes = []
if raw_info["ignored_after_cutoff"]:
    notes.append(f"2026년 9월 이후 원본 입항기록 {raw_info['ignored_after_cutoff']:,}건은 학습에서 제외합니다.")
if raw_info["ignored_before_start"]:
    notes.append(f"2013년 이전 원본 입항기록 {raw_info['ignored_before_start']:,}건은 학습에서 제외합니다.")
if raw_info["duplicate_count"]:
    notes.append(f"완전히 동일한 원본 행 {raw_info['duplicate_count']:,}개 중복을 확인했습니다.")
if notes:
    st.caption(" · ".join(notes))

if absent_passenger:
    not_approved = [m for m in absent_passenger if m not in set(confirmed_zeros)]
    if not_approved:
        st.warning("승객 학습의 미확인 무기록 월: " + ", ".join(not_approved) + " · 왼쪽에서 정말 0건인지 확인한 월만 선택해주세요.")

if schedule_bytes:
    try:
        parsed_schedule = parse_schedule_cached(schedule_bytes)
        in_year = parsed_schedule.loc[parsed_schedule["연월"].between("2027-01", "2027-12")]
        st.caption(f"2027년 스케줄 확인: {len(in_year):,}행 / 관측월 {in_year['연월'].nunique()}개월. 미기록월은 기본적으로 미확인 처리합니다.")
    except Exception as exc:
        st.error(f"예정 스케줄 확인이 필요합니다: {exc}")
        st.stop()
else:
    parsed_schedule = None

fingerprint = hashlib.sha256(
    original_bytes
    + (schedule_bytes or b"")
    + str((sorted(confirmed_zeros), remove_duplicates, complete_schedule)).encode("utf-8")
).hexdigest()
ready = confirmed_complete and (not (set(absent_passenger) - set(confirmed_zeros)))
btn = st.button("2027년 ETS 예측 및 스케줄 비교 실행", type="primary", use_container_width=True, disabled=not ready)
if not confirmed_complete:
    st.info("왼쪽에서 '2026년 9월까지 실적 수집이 완료'를 확인하면 계산 버튼이 활성화됩니다.")
elif not ready:
    st.info("승객수 학습자료에 미확인 무기록 월이 있어 실행하지 않습니다. 자료 확인 후 다시 시도해주세요.")

if btn:
    try:
        with st.status("2027년 전망 계산 중", expanded=True, state="running") as status:
            st.write("1/4 · 입항수 월별 집계 및 데이터 완결성 확인")
            m_arrival, absent_count = arrival_monthly(rows)
            st.write(f"입항수: {len(m_arrival)}개월 집계 완료 · 기록 없는 월 {len(absent_count)}개는 notebook 기준에 따라 0건 처리")
            st.write("2/4 · 승객수 월별 집계 및 이전 관측 기반 결측 보완")
            m_passenger, impute_log = passenger_monthly(rows, confirmed_zeros)
            st.write("3/4 · ETS(A,N,A) · ETS(A,Ad,A) 모델 학습 + 15개월 연속 예측")
            a_future, a_meta = forecast_arrivals_cached(m_arrival)
            p_future, p_meta = forecast_passengers_cached(m_passenger)
            st.write("4/4 · 2027년 1~12월만 추출하고 예정 스케줄과 비교")
            p_table, a_table, compare_info = compare_schedule(
                p_future, a_future, parsed_schedule, complete_schedule=complete_schedule
            )
            result = {
                "arrival_monthly": m_arrival, "passenger_monthly": m_passenger,
                "arrival_forecast": a_future, "passenger_forecast": p_future,
                "arrival_metadata": a_meta, "passenger_metadata": p_meta,
                "passenger_table": p_table, "arrival_table": a_table,
                "comparison_metadata": compare_info, "imputation_log": impute_log,
                "absent_arrival_months": absent_count,
            }
            st.session_state["dashboard_result"] = result
            st.session_state["dashboard_fingerprint"] = fingerprint
            status.update(label="예측·비교 계산 완료", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"계산을 완료할 수 없습니다: {exc}")
        st.stop()

result = st.session_state.get("dashboard_result")
if not result or st.session_state.get("dashboard_fingerprint") != fingerprint:
    st.markdown("### 자료 분석 실행 안내")
    st.markdown("""<div class="stage">두 학습기간과 모델은 고정되어 있습니다. 원자료 검증 후 실행하면
    <b>2027년 12개월 예측 그래프 2개, 월별 비교표 2개, CSV 내려받기</b>가 표시됩니다.</div>""", unsafe_allow_html=True)
    with st.expander("시뮬레이션 영상 미리보기", expanded=True):
        show_video()
    st.stop()

p_forecast, a_forecast = result["passenger_forecast"], result["arrival_forecast"]
p_table, a_table = result["passenger_table"], result["arrival_table"]
info = result["comparison_metadata"]
pred_total_passengers = float(p_forecast["예측승객수"].sum())
pred_total_arrivals = float(a_forecast["예측입항수"].sum())
known_calls = float(a_table["예정입항건수합계"].sum()) if parsed_schedule is not None else np.nan
known_capacity = float(p_table["예정 승객수"].sum()) if parsed_schedule is not None else np.nan

st.markdown("### 2027년 핵심 전망")
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    st.metric("예측 입항수 · 2027 합계", fmt_number(pred_total_arrivals, "건"))
with k2:
    st.metric("예측 승객수 · 2027 합계", fmt_number(pred_total_passengers, "명"))
with k3:
    st.metric("입항 예정 스케줄 · 확인 건수", fmt_number(known_calls, "건"))
with k4:
    st.metric("예정 승객수 · 일정 기준 합계", fmt_number(known_capacity, "명"))
if parsed_schedule is not None:
    st.caption("※ 예정 승객수는 원본 CSV의 승객정원수 합계에서 계산한 참고값입니다. 실제 승객수나 정원 확인 완료값이 아닙니다.")
if parsed_schedule is not None and info["observed_schedule_months"] < 12 and not complete_schedule:
    st.caption(f"예정 스케줄은 {info['observed_schedule_months']}/12개월만 기록되어 있습니다. 해당 합계는 현재 업로드된 행의 합이며 2027년 전체 확정값이 아닙니다.")

vtab, ttab, stab = st.tabs(["📈 2027년 예측 그래프", "📋 2027년 월별 비교표", "🎬 시뮬레이션 과정·지표 설명"])
with vtab:
    st.caption("실제(회색): 2025.01~2026.09 확정 실적 · 예측(청록): 2027.01~12만 표시 · 음영: 95% 근사 예측구간")
    st.markdown("#### 입항수 · ETS(A,N,A)")
    st.caption("학습기간: 2013년 1월~2026년 9월(165개월). 전체 기록 기간을 사용하여 장기 시계열을 유지합니다.")
    st.plotly_chart(plot_forecast(
        a_forecast, result["arrival_monthly"], "실제입항수", "입항 (건)", "ETS(A,N,A)",
        a_table["예정입항건수합계"], "2027 입항 예정 건수 (실적 아님)",
    ), use_container_width=True, key="arrival_chart")
    st.markdown("#### 승객수 · ETS(A,Ad,A)")
    st.caption("학습기간: 2023년 1월~2026년 9월(45개월). 최근 승객 자료의 연속성과 기록 완결성에 초점을 둡니다.")
    st.plotly_chart(plot_forecast(
        p_forecast, result["passenger_monthly"], "실제승객수", "승객 (명)", "ETS(A,Ad,A)",
        p_table["예정 승객수"], "2027 예정 승객수 (일정상 정원 기반)",
    ), use_container_width=True, key="passenger_chart")
    st.info("2026년 10~12월은 연속 예측의 중간 단계로만 계산하고 그래프·표에는 공개하지 않습니다. 2027년의 실제 실적은 아직 없으므로 미래 일정은 '예정'으로 구분해 표시합니다.")
    st.caption("※ 그래프의 '예정 승객수'는 일정에 기재된 승객정원수 합계로, 실제 탑승객 수나 검증된 실제 정원 확정치를 의미하지 않습니다.")
    with st.expander("모형별 연간 95% 근사 예측구간"):
        ac = result["arrival_metadata"]["annual_interval"]
        pc = result["passenger_metadata"]["annual_interval"]
        st.write(f"입항수 연간 근사 구간: **{ac[0]:,.1f} ~ {ac[1]:,.1f}건**")
        st.write(f"승객수 연간 근사 구간: **{pc[0]:,.1f} ~ {pc[1]:,.1f}명**")
        st.caption("연간 구간은 미래 경로별 12개월 합계를 먼저 산출하고 2.5%, 97.5% 분위수로 계산합니다. 월별 구간 하한·상한의 단순 합이 아닙니다.")

with ttab:
    st.markdown("#### 2027년 월별 예측 승객수·예정 승객수 비교")
    st.caption("차이 = 예정 승객수 − 예측승객수. 예정대비절대차이율 = |차이| ÷ 예정 승객수 × 100 (0명·미확인 월은 계산 불가).")
    st.dataframe(p_table.round(2), hide_index=True, use_container_width=True, height=463)
    download_csv("승객수 2027 비교표 CSV 내려받기", p_table, "2027_passenger_comparison.csv", "passenger_csv")
    st.caption("※ '예정 승객수'는 스케줄에 기재된 '승객정원수'를 월별로 합산한 참고값입니다. 실제 탑승 승객수도 아니고, 선박의 실제 정원이 확정·검증됐다는 의미도 아닙니다.")
    st.markdown("#### 2027년 월별 입항수 전망·예정 건수 비교")
    st.caption("차이 = 예측입항수 − 예정입항건수합계(입항 notebook과 동일). 절대차이율 = |차이| ÷ 예정입항건수합계 × 100. 입항 건수의 분모는 승객 정원이 아닌 예정 건수입니다.")
    st.dataframe(a_table.round(2), hide_index=True, use_container_width=True, height=463)
    download_csv("입항수 2027 비교표 CSV 내려받기", a_table, "2027_arrival_comparison.csv", "arrival_csv")
    st.caption("'실제 예정 입항수'는 업로드된 일정의 예정 입항 건수와 동일한 값으로, 요청 형식을 따라 중복 표기했습니다. 이는 실제 완료된 2027년 입항 실적이 아닙니다.")
    if info["outside_period"]:
        st.warning("스케줄에는 분석 대상 밖의 월도 있습니다: " + ", ".join(info["outside_period"]) + ". 2027년 1~12월만 비교했습니다.")
    st.info("2027년 예정 승객수(스케줄상 정원 기반) 또는 예정 입항 건수와의 차이율은 정확도 지표(MAPE·WAPE)가 아니라 '예정 스케줄 대비 편차'입니다.")

with stab:
    st.markdown("#### 자료 업로드부터 결과 출력까지")
    show_video()
    st.markdown("""<div class="stage">
        <b>실제 처리 순서</b> &nbsp; ① 원자료·일정 CSV 읽기 → ② 컬럼/중복/기간/결측 검증 →
        ③ 입항수 ETS(A,N,A)와 승객수 ETS(A,Ad,A) 적합 →
        ④ 2026.10~2027.12 연속 예측·근사구간 시뮬레이션 →
        ⑤ 2027년 12개월만 추출 → ⑥ 예정 스케줄과 대조하고 그래프·표 생성
        </div>""", unsafe_allow_html=True)
    st.markdown("#### 평가 지표 정의")
    st.markdown("""
    - **오차** = 실제 − 예측. 양수면 실제가 예측보다 큽니다.
    - **MAPE(%)** = `mean(|실제−예측| / 실제 × 100)`. 실제가 **0 또는 결측인 월은 제외**합니다.
    - **WAPE(%)** = `Σ|실제−예측| / Σ실제 × 100`. 유효월 실제 합계가 0이면 계산하지 않습니다.
    - **95% 근사 예측구간 포함률(%)** = `구간 안에 실제가 있는 유효월 수 / 구간이 존재하는 실제 유효월 수 × 100`.
    """)
    st.caption("2027년 실제 관측값이 없어 현재는 2027년 MAPE·WAPE·포함률을 계산하지 않습니다. 공식은 향후 실제 실적 확보 후 사후 검증에 사용할 수 있습니다.")
    st.markdown("#### 데이터 해석 및 재현 기준")
    st.write("**입항수:** 입항 기록만 선택, 2013.01~2026.09 매월 합산, 무기록 월 0건 처리(첨부 노트북 기준). 계절 주기 12, 초기화 기본 estimated, 1,000개 미래 경로(잔차 복원추출, 시드 20261029).")
    st.write("**승객수:** 2023.01~2026.09 입항 승객합계. 무입항으로 확인된 무기록 월만 0명 처리. 승객 결측은 과거 동일 호출부호 관측의 중앙값으로 학습 데이터에만 보완합니다. 계절 주기 12, heuristic 초기화, 천 명 단위 학습, 10,000개 미래 경로(시드 20261030).")
    st.write("**다른 기간을 사용하는 이유:** 입항 기록은 장기간의 패턴을 활용할 수 있지만, 승객 수는 최근 기간의 연속된 학습용 기록을 기반으로 안정적으로 집계하려는 목적입니다. 이는 모델 설정상의 판단이지 두 데이터의 품질이 동일하다는 의미는 아닙니다.")
    st.warning("95% 구간은 모수 추정 및 자료 결측/보완, 장래의 선박 일정 변경 등 모든 불확실성을 포괄하지 않는 근사 예측구간입니다. 실제값 95% 포함을 보장하지 않습니다.")
    st.markdown("#### 재현성 및 자료 점검")
    st.write(f"입항수: **{result['arrival_metadata']['periods']}개월** · 승객수: **{result['passenger_metadata']['periods']}개월** · 결과는 2027년 **12개월**")
    if result["absent_arrival_months"]:
        st.caption(f"입항수에서 무기록을 0건으로 처리한 월: {len(result['absent_arrival_months'])}개. 원본 누락이 아닌지 확인이 필요합니다.")
    if not result["imputation_log"].empty:
        with st.expander("승객 결측 보완 내역 확인"):
            st.dataframe(result["imputation_log"], hide_index=True, use_container_width=True)
    with st.expander("실제 사용한 승객 학습자료 보기"):
        st.dataframe(result["passenger_monthly"], hide_index=True, use_container_width=True)
    st.caption("원자료는 실행 중 처리하며 외부 API나 별도의 사용자 데이터베이스로 전송하도록 구현하지 않았습니다. Streamlit Community Cloud에 배포할 때는 운영 플랫폼 자체의 데이터 처리 정책이 적용됩니다.")
