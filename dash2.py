"""부산항 크루즈 입항/승객 월별 갱신형 ETS — Streamlit 웹 앱."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Launch correctly when PyCharm's ordinary Run button executes this file.
# Without a Streamlit runner context, st.stop() is not a reliable exit.
import subprocess
import sys
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    if __name__ == "__main__":
        entry = Path(__file__).resolve()
        command = [sys.executable, "-m", "streamlit", "run", str(entry)]
        print("Starting the Streamlit dashboard: " + " ".join(command), flush=True)
        try:
            raise SystemExit(subprocess.call(command, cwd=str(entry.parent)))
        except KeyboardInterrupt:
            raise SystemExit(0)
    raise RuntimeError("Use 'python -m streamlit run <script.py>' for this dashboard")

from engine import (
    ARRIVAL_START, PASSENGER_START, CUTOFF as BASE_CUTOFF, KNOWN_ZERO,
    month_labels, read_csv_auto, load_arrival_rows, missing_passenger_months,
    new_arrival_absences, arrival_monthly, passenger_monthly,
    forecast_arrivals, forecast_passengers, load_schedule, compare_schedule,
)
from history import create_snapshot, parse_snapshot, snapshot_bytes, evaluate_snapshot

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="부산항 크루즈 | 갱신형 ETS 전망",
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
/* 사이드바: 좁은 252px 유지 + 밝은 배경/검은 글씨로 대비 개선 */
[data-testid="stSidebar"] {
  background:#edf4f9 !important;
  color:#17283b !important;
  border-right:1px solid #d6e3ed;
  width:252px !important; min-width:252px !important; max-width:252px !important;
  flex:0 0 252px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-left:.85rem; padding-right:.85rem;
}
/* 파일 업로드, 선택 상자, 체크박스, 캡션 등 sidebar 내부 텍스트 */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color:#17283b !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] small {
  color:#405870 !important;
}
/* 업로드 드롭존과 텍스트 가시성 */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background:#ffffff !important;
  border:1px dashed #9eb5c8 !important;
  border-radius:11px;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
  color:#17283b !important;
}
/* Streamlit/Baseweb 드롭다운은 입력 표면을 흰색으로, 값은 검은색으로 */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
  background-color:#ffffff !important;
  color:#17283b !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="tag"] * {
  color:#17283b !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background:#d4f3ef !important;
}
[data-testid="stSidebar"] input::placeholder {
  color:#5a6f82 !important;
  opacity:1;
}
/* 사이드바 버튼들도 검은 글씨 + 연한 배경, 실행 버튼은 연청록 */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  color:#17283b !important;
  background:#ffffff !important;
  border:1px solid #afc4d5 !important;
  border-radius:9px;
}
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
  color:#103c42 !important;
  background:#b9ece7 !important;
  border-color:#42a8a6 !important;
  font-weight:700;
}
[data-testid="stSidebar"] .stButton button:disabled {
  opacity:.60;
}
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
/* 독립된 소제목 + 내용 단위를 스크롤할 때 부드럽게 화면 시작에 정렬 */
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"], section.main {
  scroll-snap-type:y proximity;
  scroll-padding-top:4.3rem;
}
.st-key-page_arrivals_chart,
.st-key-page_passengers_chart,
.st-key-page_passengers_table,
.st-key-page_arrivals_table,
.st-key-page_workflow,
.st-key-page_explainer,
.st-key-page_missing_notes,
.st-key-page_history_save, .st-key-page_history_evaluate {
  scroll-snap-align:start;
  scroll-snap-stop:normal;
  scroll-margin-top:1rem;
  padding:1.25rem 1.6rem 1.4rem;
  border:1px solid #e0eaf0;
  border-radius:18px;
  background:#fff;
  box-shadow:0 5px 22px rgba(27,64,91,.045);
  margin:.5rem 0 1.5rem;
}
.st-key-page_arrivals_chart,
.st-key-page_passengers_chart,
.st-key-page_passengers_table,
.st-key-page_arrivals_table {
  min-height:calc(100dvh - 7rem);
}
/* 상단 소제목으로 그래프/표의 시작이 명확하도록 */
.section-kicker{font-size:11px;font-weight:800;color:#138d9c;letter-spacing:.08em;margin-bottom:.2rem;}
.section-caption{font-size:13px;color:#637a90;line-height:1.75;margin:0 0 .5rem;}
@media (max-width:900px){
 [data-testid="stSidebar"]{width:240px !important;min-width:240px !important;max-width:240px !important;flex-basis:240px !important;}
 .st-key-page_arrivals_chart,.st-key-page_passengers_chart,
 .st-key-page_passengers_table,.st-key-page_arrivals_table {min-height:auto;}
}
@media (max-width:700px){
 .block-container{padding:1.3rem 1rem 3rem;} .hero{padding:22px 20px;}
 .hero .headline{font-size:24px;} .intro-card{min-height:unset;}
 .st-key-page_arrivals_chart,.st-key-page_passengers_chart,
 .st-key-page_passengers_table,.st-key-page_arrivals_table,
 .st-key-page_workflow,.st-key-page_explainer,.st-key-page_missing_notes,
.st-key-page_history_save, .st-key-page_history_evaluate {padding:1rem;}
 [data-testid="stSidebar"]{width:min(260px,85vw) !important;min-width:0 !important;max-width:85vw !important;flex-basis:min(260px,85vw) !important;}
}
@media (prefers-reduced-motion:reduce){
 [data-testid="stAppViewContainer"] > .main,
 [data-testid="stMain"],section.main {scroll-behavior:auto;}
}
</style>
""", unsafe_allow_html=True)



def fmt_number(value, unit="", digits=0):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}{unit}"


def file_template(name: str) -> bytes:
    return (ROOT / "templates" / name).read_bytes()


@st.cache_data(show_spinner=False)
def parse_original_cached(content: bytes, remove_duplicates: bool, cutoff: str):
    return load_arrival_rows(content, remove_duplicates=remove_duplicates, cutoff=cutoff)


@st.cache_data(show_spinner=False)
def forecast_arrivals_cached(monthly: pd.DataFrame, cutoff: str, year: int):
    return forecast_arrivals(monthly, cutoff=cutoff, target_year=year)


@st.cache_data(show_spinner=False)
def forecast_passengers_cached(monthly: pd.DataFrame, cutoff: str, year: int):
    return forecast_passengers(monthly, cutoff=cutoff, target_year=year)


@st.cache_data(show_spinner=False)
def parse_schedule_cached(content: bytes):
    return load_schedule(content)


@st.cache_data(show_spinner=False)
def inspect_raw_months(content: bytes) -> tuple[str, str, str]:
    """Read last observed arrival date without presuming that it is a completed month."""
    frame = read_csv_auto(content)
    if not {"입출", "입항일시"}.issubset(frame.columns):
        raise ValueError("원자료에 입출 또는 입항일시 열이 없습니다.")
    observed = frame.loc[frame["입출"].astype(str).str.strip().eq("입항"), "입항일시"]
    parsed = pd.to_datetime(observed, errors="coerce")
    if parsed.empty or parsed.isna().any():
        raise ValueError("입항일시 열을 확인해주세요.")
    return parsed.min().strftime("%Y-%m"), parsed.max().strftime("%Y-%m"), parsed.max().strftime("%Y-%m-%d")


def show_video():
    video = ROOT / "assets" / "workflow_demo.mp4"
    if video.is_file():
        st.video(str(video))
        st.caption("기존 처리 흐름을 설명하는 애니메이션입니다. 영상 속 수치는 실제 예측 결과가 아닙니다.")
    else:
        st.info("애니메이션 파일이 없는 배포본입니다.")


def download_csv(label, frame: pd.DataFrame, filename: str, key: str):
    st.download_button(label, data=frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=key)


def plot_forecast(future: pd.DataFrame, observed: pd.DataFrame, value_col: str, unit: str,
                  cutoff: str, year: int, model: str, plan: pd.Series | None = None,
                  plan_name: str = "예정 스케줄", missing_actual_months: list[str] | None = None):
    """Same gray / turquoise / band style, with a clearly labelled compressed time gap.

    Categorical months avoid years of empty plot space when 2030+ is selected.
    Separate traces never join actuals to predictions. Red X is in paper coordinates,
    not a numerical passenger count.
    """
    start = str(max(pd.Period(cutoff, freq="M") - 20, pd.Period(observed["연월"].iloc[0], freq="M")))
    hist = observed.loc[observed["연월"].between(start, cutoff)].copy()
    past_months = hist["연월"].tolist()
    future_months = future["연월"].tolist()
    gap = "···  중간 예측기간 화면 생략  ···"
    categories = past_months + [gap] + future_months
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_months, y=future["95% 상한"], mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="95% 상한"
    ))
    fig.add_trace(go.Scatter(
        x=future_months, y=future["95% 하한"], mode="lines", fill="tonexty",
        fillcolor="rgba(19,151,163,0.17)", line=dict(width=0),
        name="95% 근사 예측구간", hovertemplate="%{x}<br>95% 하한 %{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=past_months, y=hist[value_col], mode="lines+markers",
        name="확인 가능한 과거 실적", line=dict(color="#8f9db1", width=2.6),
        marker=dict(size=6), hovertemplate="%{x}<br>실제: %{y:,.0f}<extra></extra>"
    ))
    pred_col = "예측승객수" if value_col == "실제승객수" else "예측입항수"
    fig.add_trace(go.Scatter(
        x=future_months, y=future[pred_col], mode="lines+markers",
        name=f"선택 연도 예측 · {model}", line=dict(color="#0b8793", width=3.5),
        marker=dict(size=7, color="#0b8793"),
        hovertemplate="%{x}<br>예측: %{y:,.2f}<extra></extra>"
    ))
    if plan is not None and plan.notna().any():
        fig.add_trace(go.Scatter(
            x=future_months, y=plan, mode="lines+markers", name=plan_name,
            line=dict(color="#f4a456", width=2.4, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x}<br>예정: %{y:,.0f}<extra></extra>"
        ))
    for month in (missing_actual_months or []):
        if month in past_months:
            fig.add_shape(
                type="line", xref="x", yref="paper", x0=month, x1=month, y0=0, y1=0.88,
                line=dict(color="rgba(211,47,47,0.38)", width=1.4, dash="dot"), layer="below"
            )
            fig.add_annotation(
                x=month, y=.96, xref="x", yref="paper", text="<b>✕</b>",
                showarrow=False, font=dict(size=24, color="#d32f2f"),
                bgcolor="rgba(255,255,255,0.90)", borderpad=1,
            )
            fig.add_annotation(
                x=month, y=.875, xref="x", yref="paper", text="결측", showarrow=False,
                font=dict(size=10, color="#b42318"), bgcolor="rgba(255,255,255,0.90)", borderpad=1,
            )
    ticks = [past_months[i] for i in range(0, len(past_months), 4)] + [gap] + future_months[::2]
    # Ensure last month is visible even if not captured by step.
    if future_months[-1] not in ticks:
        ticks.append(future_months[-1])
    ticktext = ["생략 구간" if x == gap else x.replace("-", ".") for x in ticks]
    fig.update_layout(
        height=430, margin=dict(l=16, r=16, t=31, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="Noto Sans KR, Malgun Gothic, sans-serif", color="#38536d", size=12),
        hovermode="x unified", legend=dict(orientation="h", y=-.27, x=0, font=dict(size=11)),
        xaxis=dict(type="category", categoryorder="array", categoryarray=categories,
                   tickmode="array", tickvals=ticks, ticktext=ticktext,
                   showgrid=True, gridcolor="#edf2f7", linecolor="#d5e1ea"),
        yaxis=dict(title=unit, rangemode="tozero", showgrid=True,
                   gridcolor="#edf2f7", zeroline=False),
    )
    return fig


# === Header / original four-section scrolling visual language ===
st.markdown("""
<div class="hero">
  <div class="eyebrow">BUSAN PORT · CRUISE INTELLIGENCE</div>
  <div class="headline">크루즈 운영 전망 · 월별 갱신형 ETS</div>
  <div class="subtitle">새로 확정된 실적을 반영해 모델을 재학습하고, 선택한 연도의 입항수·승객수를 전망합니다.<br>
  실제 실적과 예정 스케줄을 구분하며 월별 95% 근사 예측구간을 제시합니다.</div>
  <span class="pill">ETS 모델 유지</span><span class="pill">최신 확정월까지 재학습</span>
  <span class="pill">예측 연도 선택</span><span class="pill">과거 예측 기록 보관</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("데이터 설정")
    st.caption("STEP 01 · 원자료 / 예정 스케줄")
    raw_upload = st.file_uploader("누적 입출항 원자료 CSV", type=["csv"],
                                  help="2013년부터 최신 기록까지 하나의 누적 CSV로 업로드하세요.")
    schedule_upload = st.file_uploader("선택 연도 예정 스케줄 CSV (선택)", type=["csv"],
                                       help="입항예정일, 선명, 승객정원수 열. 연도를 선택하면 그해 일정만 비교합니다.")
    with st.expander("CSV 헤더 양식"):
        st.download_button("원자료 헤더", file_template("raw_template.csv"), "raw_template.csv", "text/csv", key="template_raw")
        st.download_button("예정 일정 헤더", file_template("schedule_template.csv"), "schedule_template.csv", "text/csv", key="template_schedule")
    remove_duplicates = st.checkbox("원자료의 완전히 동일한 행 중복 제거", value=False)

raw_bytes = raw_upload.getvalue() if raw_upload is not None else None
schedule_bytes = schedule_upload.getvalue() if schedule_upload is not None else None

if raw_bytes is None:
    one, two, three = st.columns(3, gap="medium")
    with one:
        st.markdown('''<div class="intro-card"><div class="cap">ARRIVALS</div><div class="big">ETS(A,N,A)</div><div class="small">최초 학습: 2013.01~2026.09<br>이후 확인된 실적을 계속 추가</div></div>''', unsafe_allow_html=True)
    with two:
        st.markdown('''<div class="intro-card"><div class="cap">PASSENGERS</div><div class="big">ETS(A,Ad,A)</div><div class="small">최초 학습: 2023.01~2026.09<br>승객 결측은 학습용에만 별도 보완</div></div>''', unsafe_allow_html=True)
    with three:
        st.markdown('''<div class="intro-card"><div class="cap">FORECAST</div><div class="big">연도 선택</div><div class="small">한 번의 모델 적합으로 선택 연도까지 연속 예측<br>화면에는 해당 연도 12개월만 표시</div></div>''', unsafe_allow_html=True)
    st.markdown("### 업로드 후 진행되는 절차")
    st.markdown('<div class="stage">① 실적 완료월 선택·확인 → ② 결측·무입항 월 검토 → ③ ETS 모델 재학습 → ④ 선택 연도 전망 → ⑤ 예측 기록 저장·추후 실적 평가</div>', unsafe_allow_html=True)
    show_video()
    st.info("먼저 왼쪽 사이드바에서 누적 입출항 원자료를 업로드해주세요. 입력 자료를 임의로 예측값에 넣지 않습니다.")
    st.stop()

try:
    first_recorded, last_recorded, last_recorded_date = inspect_raw_months(raw_bytes)
    if last_recorded < BASE_CUTOFF:
        st.error("누적 자료에 최소 2026년 9월까지의 기록이 필요합니다.")
        st.stop()
except Exception as exc:
    st.error(f"원자료 날짜 확인 실패: {exc}")
    st.stop()

options = month_labels(BASE_CUTOFF, last_recorded)
prev_to_last = str(pd.Period(last_recorded, freq="M") - 1)
default_cutoff = max(BASE_CUTOFF, prev_to_last)
with st.sidebar:
    st.divider()
    st.caption("STEP 02 · 확정된 월까지 학습")
    cutoff = st.selectbox("실적 수집이 완료된 마지막 월", options=options,
                          index=options.index(default_cutoff),
                          help="마지막 기록월은 월중 자료일 수 있으므로 자동 확정하지 않습니다.")
    confirmed_complete = st.checkbox(
        f"{cutoff}까지 실제 입항 자료의 수집·마감을 확인했습니다",
        value=False, key="confirmed_" + hashlib.sha256(raw_bytes).hexdigest()[:12] + cutoff,
    )
    st.caption(f"원자료 마지막 입항 기록: {last_recorded_date} · 마지막 기록월은 자동으로 완료월로 확정하지 않습니다.")
    st.divider()
    st.caption("STEP 03 · 예측 연도")
    future_years = list(range(int(cutoff[:4]) + 1, int(cutoff[:4]) + 12))
    selected_year = st.selectbox("예측할 연도", options=future_years,
                                 format_func=lambda y: f"{y}년 (1~12월)",
                                 help="마지막 확정 실적 연도의 다음 연도부터 선택 가능합니다.")
    st.caption("현재 선택한 연도의 12개월만 그래프·비교표에 표시됩니다.")

try:
    rows, raw_info = parse_original_cached(raw_bytes, remove_duplicates, cutoff)
except Exception as exc:
    st.error(f"원자료 검증 실패: {exc}")
    st.stop()

absent_passenger = missing_passenger_months(rows, cutoff)
absent_recent_arrivals = new_arrival_absences(rows, cutoff)
all_absent = sorted(set(absent_passenger) | set(absent_recent_arrivals))
with st.sidebar:
    st.divider()
    st.caption("STEP 04 · 무입항 월 검증")
    confirmed_zeros = st.multiselect(
        "무입항(0건)이 확인된 기록 없는 월", options=all_absent,
        default=[m for m in all_absent if m in KNOWN_ZERO],
        help="승객수 학습 및 기준선 이후 입항수에는 확인된 무입항 월만 0으로 반영합니다.",
    )
    st.caption("2023-01·02·12, 2024-01은 이전 팀 확인 월로 초기 선택합니다. 새 무기록 월은 반드시 확인해주세요.")
    complete_schedule = st.checkbox(
        f"{selected_year}년 예정 스케줄이 12개월 전체를 포괄합니다", value=False,
        help="체크한 경우에만 스케줄에 없는 월을 예정 0건·0명으로 해석합니다.",
    )

unapproved = sorted(set(all_absent) - set(confirmed_zeros))
if unapproved:
    st.warning("아직 무입항 여부가 확인되지 않은 월: " + ", ".join(unapproved) + ". 검증 없이 임의로 0으로 바꾸지 않습니다.")
if raw_info["ignored_after_cutoff"]:
    st.caption(f"학습기간 이후의 입항 기록 {raw_info['ignored_after_cutoff']:,}건은 아직 학습에 포함하지 않았습니다.")
if cutoff == last_recorded:
    st.warning("마지막 입항 기록월을 완료월로 선택했습니다. 그 달의 전체 자료가 실제로 확정되었는지 확인하세요. 마지막 입항일만으로 월 마감 여부를 알 수 없습니다.")

if schedule_bytes:
    try:
        schedule = parse_schedule_cached(schedule_bytes)
        sched_year = schedule.loc[schedule["연월"].between(f"{selected_year}-01", f"{selected_year}-12")]
        st.caption(f"{selected_year}년 예정 스케줄: {len(sched_year):,}행 · 기록 월 {sched_year['연월'].nunique()}개월")
        if sched_year.empty:
            st.warning(f"선택한 {selected_year}년의 일정이 업로드된 스케줄에 없습니다. 비교표의 예정값은 미확인으로 표시합니다.")
    except Exception as exc:
        st.error(f"예정 스케줄 검증 실패: {exc}")
        st.stop()
else:
    schedule = None

one, two, three = st.columns(3, gap="medium")
with one:
    st.markdown(f'''<div class="intro-card"><div class="cap">ARRIVALS · 추세 미적용 계절모형</div><div class="big">ETS(A,N,A)</div><div class="small">학습: 2013.01 ~ {cutoff.replace('-', '.')} ({len(month_labels(ARRIVAL_START,cutoff))}개월)<br>한 번 적합한 모형의 다이렉트 예측만 사용</div></div>''', unsafe_allow_html=True)
with two:
    st.markdown(f'''<div class="intro-card"><div class="cap">PASSENGERS · 감쇠 추세 계절모형</div><div class="big">ETS(A,Ad,A)</div><div class="small">학습: 2023.01 ~ {cutoff.replace('-', '.')} ({len(month_labels(PASSENGER_START,cutoff))}개월)<br>원자료 결측과 학습용 보완 구분</div></div>''', unsafe_allow_html=True)
with three:
    st.markdown(f'''<div class="intro-card"><div class="cap">FORECAST · 선택한 연도</div><div class="big">{selected_year}년 12개월</div><div class="small">{str(pd.Period(cutoff,freq="M")+1)}부터 {selected_year}.12까지 연속 예측<br>{selected_year}년 1~12월만 화면 표시</div></div>''', unsafe_allow_html=True)

fingerprint = hashlib.sha256(
    raw_bytes + (schedule_bytes or b"")
    + str((cutoff, selected_year, sorted(confirmed_zeros), remove_duplicates, complete_schedule)).encode()
).hexdigest()
ready = confirmed_complete and not unapproved
with st.sidebar:
    st.divider()
    st.caption("STEP 05 · 재학습 / 전망")
    compute = st.button(f"{selected_year}년 ETS 재학습·예측 실행", type="primary",
                        use_container_width=True, disabled=not ready)
    st.caption("예측을 갱신하면 새로운 예측 기록이 만들어집니다. 이전 예측은 JSON으로 저장하세요.")

if not confirmed_complete:
    st.info("왼쪽에서 확정된 실적 완료월을 확인하면 계산 버튼이 활성화됩니다.")
elif unapproved:
    st.info("기록이 없는 월을 확인하거나 수정된 원자료 CSV를 업로드하면 계산할 수 있습니다.")

if compute:
    try:
        with st.status("월별 데이터 검증 · ETS 재학습 · 예측구간 계산", expanded=True) as status:
            st.write("1/4 · 선택 완료월까지 월별 실적 집계")
            a_monthly, absent_arrival_months = arrival_monthly(rows, cutoff, confirmed_zeros)
            p_monthly, impute_log = passenger_monthly(rows, confirmed_zeros, cutoff)
            st.write(f"2/4 · 입항수 {len(a_monthly)}개월, 승객수 {len(p_monthly)}개월 학습")
            a_fc, a_meta = forecast_arrivals_cached(a_monthly, cutoff, selected_year)
            st.write("3/4 · 승객수 ETS(A,Ad,A) 학습 및 근사 예측구간")
            p_fc, p_meta = forecast_passengers_cached(p_monthly, cutoff, selected_year)
            st.write("4/4 · 선택 연도 12개월 추출, 일정 비교, 기록 생성")
            p_table, a_table, info = compare_schedule(
                p_fc, a_fc, schedule, complete_schedule=complete_schedule, target_year=selected_year
            )
            snap = create_snapshot(cutoff, selected_year, a_fc, p_fc, a_meta, p_meta, raw_bytes)
            st.session_state["dashboard_result"] = {
                "a_monthly": a_monthly, "p_monthly": p_monthly,
                "a_fc": a_fc, "p_fc": p_fc, "a_meta": a_meta, "p_meta": p_meta,
                "p_table": p_table, "a_table": a_table, "info": info,
                "impute_log": impute_log, "absent_arrival_months": absent_arrival_months,
                "snapshot": snap,
            }
            st.session_state["dashboard_fingerprint"] = fingerprint
            status.update(label="재학습·예측 완료", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"예측을 계산할 수 없습니다: {exc}")
        st.stop()

result = st.session_state.get("dashboard_result")
if result is None or st.session_state.get("dashboard_fingerprint") != fingerprint:
    st.markdown("### 최신 실적을 이용한 ETS 예측")
    st.markdown('''<div class="stage">왼쪽에서 실적 완료월과 새 무입항 월을 확인하고 실행하면, <b>승객수 예측 → 입항수 예측 → 결측 설명</b> 순서로 섹션별 그래프가 표시됩니다. 기존 결과는 예측 기록 JSON으로 보관할 수 있습니다.</div>''', unsafe_allow_html=True)
    with st.expander("시뮬레이션 과정 애니메이션", expanded=False):
        show_video()
    st.stop()

p_forecast, a_forecast = result["p_fc"], result["a_fc"]
p_table, a_table = result["p_table"], result["a_table"]
info = result["info"]
known_calls = float(a_table["예정입항건수합계"].sum(min_count=1)) if schedule is not None else np.nan
known_capacity = float(p_table["예정 승객수"].sum(min_count=1)) if schedule is not None else np.nan

st.markdown(f"### {selected_year}년 핵심 전망")
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    st.metric("예측 입항수 합계", fmt_number(a_forecast["예측입항수"].sum(), "건", 2))
with k2:
    st.metric("예측 승객수 합계", fmt_number(p_forecast["예측승객수"].sum(), "명"))
with k3:
    st.metric("예정 입항 건수 (업로드 일정)", fmt_number(known_calls, "건"))
with k4:
    st.metric("예정 승객수 (정원 기반)", fmt_number(known_capacity, "명"))
if schedule is not None:
    st.caption("※ '예정 승객수'는 업로드 스케줄의 '승객정원수' 합계로 만든 참고값입니다. 실제 탑승 승객수가 아니며 정원이 검증된 값이라는 뜻도 아닙니다.")
if result["a_meta"]["forecast_horizon_months"] > 24:
    st.warning("최종 실적 이후 24개월을 넘는 장기 예측입니다. 일정·정책·선박 변화가 반영되지 않아 불확실성이 커집니다. 가능하면 새 실적이 확정될 때마다 재학습하세요.")

vtab, ttab, htab, mtab = st.tabs([
    f"📈 {selected_year}년 예측 그래프", f"📋 {selected_year}년 비교표",
    "🗂️ 예측 기록·실제값 평가", "🎬 시뮬레이션·방법론",
])

with vtab:
    st.caption("회색 선: 확인 가능한 최근 실제 실적 · 청록 선: 선택한 연도의 ETS 예측 · 청록 음영: 95% 근사 예측구간 · 빨간 X: 승객수가 기록되지 않은 월")
    st.info(
        f"**과거와 예측 사이의 축은 생략해 표시합니다.** 실제 학습은 {cutoff}까지이며, "
        f"{str(pd.Period(cutoff, freq='M')+1)}~{selected_year}.12를 연속 예측한 후 "
        f"{selected_year}년 12개월만 표시합니다. 중간 구간은 결측이 아니고, 학습기간 전체를 그래프에 그린 것도 아닙니다."
    )
    p_monthly = result["p_monthly"]
    missing = p_monthly.loc[p_monthly["결측입항건수"].gt(0)].copy()
    missing_labels = missing["연월"].tolist()
    with st.container(key="page_passengers_chart", border=False):
        st.markdown('<div class="section-kicker">01 / PASSENGER FORECAST</div>', unsafe_allow_html=True)
        st.subheader(f"승객수 예측 · {selected_year}년")
        st.caption(f"ETS(A,Ad,A) 감쇠 추세 계절모형 | 학습: {PASSENGER_START}~{cutoff} ({result['p_meta']['periods']}개월)")
        st.plotly_chart(plot_forecast(
            p_forecast, p_monthly, "실제승객수", "승객 (명)", cutoff, selected_year,
            "ETS(A,Ad,A)", p_table["예정 승객수"], "예정 승객수 (정원 기반)", missing_labels,
        ), use_container_width=True, key="passenger_chart")
        st.caption("빨간 X: 승객수가 누락되어 실제 월합계를 확정할 수 없는 월(0명 아님). X의 높이는 승객수 값이 아닙니다.")
        st.caption("※ 예정 승객수는 스케줄에 기재된 승객정원수의 월별 합계입니다. 실제 탑승 승객수가 아닙니다.")

    with st.container(key="page_arrivals_chart", border=False):
        st.markdown('<div class="section-kicker">02 / ARRIVAL FORECAST</div>', unsafe_allow_html=True)
        st.subheader(f"입항수 예측 · {selected_year}년")
        st.caption(f"ETS(A,N,A) 추세 미적용 계절모형 | 학습: {ARRIVAL_START}~{cutoff} ({result['a_meta']['periods']}개월) | 다이렉트 예측만 사용")
        st.plotly_chart(plot_forecast(
            a_forecast, result["a_monthly"], "실제입항수", "입항 (건)", cutoff,
            selected_year, "ETS(A,N,A)", a_table["예정입항건수합계"], "입항 예정 건수",
        ), use_container_width=True, key="arrival_chart")
        st.caption("회색: 최근 실제 입항 기록 · 청록: 선택한 연도 예측 · 음영: 95% 근사 구간 · 주황 점선: 예정 스케줄")

    with st.container(key="page_missing_notes", border=False):
        st.markdown('<div class="section-kicker">DATA QUALITY · 결측과 학습용 보완</div>', unsafe_allow_html=True)
        st.subheader("승객수 결측·무입항 월 안내")
        if not missing.empty:
            st.warning(
                f"승객수 기록 누락 **{int(missing['결측입항건수'].sum())}건** "
                f"(월: {', '.join(missing['연월'].tolist())}). 입항은 있었지만 승객수가 비어 있어 "
                "해당 월의 실제 승객수 합계를 확정할 수 없습니다. 0명으로 처리하지 않습니다."
            )
            st.caption("학습용 승객수에 한해서 해당 입항 시점 이전 같은 호출부호의 관측 중앙값으로 보완합니다. 실제 승객수는 계속 미확정으로 남깁니다.")
            if not result["impute_log"].empty:
                with st.expander("선박·입항일·학습용 보완값 확인"):
                    st.dataframe(result["impute_log"], hide_index=True, use_container_width=True)
        else:
            st.success("학습기간의 개별 입항 승객수 누락이 확인되지 않았습니다.")
        zero_months = p_monthly.loc[p_monthly["자료상태"].eq("확인된 무입항 0명"), "연월"].tolist()
        if zero_months:
            st.caption("실제 무입항으로 확인한 0명 월: " + ", ".join(zero_months))
        with st.expander("선택 연도 합계의 95% 근사 예측구간"):
            al, au = result["a_meta"]["annual_interval"]
            pl, pu = result["p_meta"]["annual_interval"]
            st.write(f"입항수 연간 근사 구간: **{al:,.1f} ~ {au:,.1f}건**")
            st.write(f"승객수 연간 근사 구간: **{pl:,.1f} ~ {pu:,.1f}명**")
            st.caption("각 시뮬레이션 경로에서 선택 연도 12개월을 합산한 뒤 2.5%, 97.5% 분위수를 구했습니다. 월별 구간의 합이 아닙니다.")

with ttab:
    with st.container(key="page_passengers_table", border=False):
        st.markdown('<div class="section-kicker">03 / PASSENGER COMPARISON</div>', unsafe_allow_html=True)
        st.subheader(f"{selected_year}년 월별 승객수 비교표")
        st.caption("차이 = 예정 승객수 − 예측승객수 · 예정대비절대차이율 = |차이| ÷ 예정 승객수 × 100")
        st.dataframe(p_table.round(2), hide_index=True, use_container_width=True, height=440)
        download_csv("승객수 비교표 CSV", p_table, f"passengers_compare_{selected_year}.csv", "p_csv")
        st.caption("※ 예정 승객수는 스케줄상 승객정원수를 합산한 참고값으로, 실제 승객수가 아닙니다.")
    with st.container(key="page_arrivals_table", border=False):
        st.markdown('<div class="section-kicker">04 / ARRIVAL COMPARISON</div>', unsafe_allow_html=True)
        st.subheader(f"{selected_year}년 월별 입항수 비교표")
        st.caption("차이 = 예측입항수 − 예정입항건수합계 · 절대차이율 = |차이| ÷ 예정입항건수합계 × 100")
        st.dataframe(a_table.round(2), hide_index=True, use_container_width=True, height=440)
        download_csv("입항수 비교표 CSV", a_table, f"arrivals_compare_{selected_year}.csv", "a_csv")
        st.caption("'실제 예정 입항수'는 실적이 아닌 예정 스케줄상의 입항 건수입니다.")
    if info["outside_period"]:
        st.caption("선택 연도 이외 일정은 비교에서 제외했습니다. 이외 연월: " + ", ".join(info["outside_period"][:18]))
    st.info("예정 스케줄과 ETS 예측값의 차이는 **계획 대비 편차**이지, MAPE·WAPE로 평가하는 실제 예측 정확도가 아닙니다.")

with htab:
    with st.container(key="page_history_save", border=False):
        st.markdown('<div class="section-kicker">05 / FORECAST ARCHIVE</div>', unsafe_allow_html=True)
        st.subheader("현재 예측 기록 보관")
        snap = result["snapshot"]
        st.caption(
            f"예측 생성: {snap['created_at_utc']} · 학습 완료월: {cutoff} · "
            f"예측 대상: {selected_year}년 · 기록 ID: {snap['record_id'][:10]}"
        )
        json_content = snapshot_bytes(snap)
        st.download_button(
            "현재 예측 기록 JSON 내려받기", data=json_content,
            file_name=f"ets_forecast_{cutoff}_{selected_year}_{snap['record_id'][:8]}.json",
            mime="application/json", use_container_width=True, key="save_snapshot_json",
        )
        st.caption("JSON에는 예측값·구간·모델·시점 정보만 저장합니다. 원자료의 선박별 기록은 포함하지 않습니다. 이후 새 실적이 들어오면 이 JSON으로 당시 예측을 평가할 수 있습니다.")
        baseline_file = ROOT / "reference" / "baseline_2026-09_2027.json"
        if baseline_file.is_file():
            st.download_button(
                "최초 2027년 기준선 예측 기록 JSON",
                data=baseline_file.read_bytes(),
                file_name="ets_baseline_2026-09_2027.json", mime="application/json",
                key="baseline_snapshot_download",
            )
            st.caption("초기 2026-09 학습자료로 복원 계산한 2027년 기준선입니다. 당시 실적값을 새로 학습한 결과로 덮어쓰지 않습니다.")
        st.caption("과거 예측은 다운로드한 JSON 파일을 직접 보관해주세요. 공개 Streamlit 서버에는 파일을 저장하지 않습니다.")

    with st.container(key="page_history_evaluate", border=False):
        st.markdown('<div class="section-kicker">06 / HISTORICAL EVALUATION</div>', unsafe_allow_html=True)
        st.subheader("이전 예측 vs 새로 확정된 실적")
        baseline_file = ROOT / "reference" / "baseline_2026-09_2027.json"
        use_baseline = st.checkbox("최초 2027년 기준선으로 실제 실적 평가", value=False,
                                   help="최초 2026-09 기준 학습 예측값을 변경하지 않고, 새 실적과 비교합니다.")
        saved_upload = st.file_uploader("저장했던 예측 기록 JSON 업로드", type=["json"], key="old_forecast_upload")
        historical_bytes = (saved_upload.getvalue() if saved_upload is not None else
                            (baseline_file.read_bytes() if use_baseline and baseline_file.is_file() else None))
        if historical_bytes:
            try:
                old_meta, old_frames = parse_snapshot(historical_bytes)
                st.caption(f"이전 예측: {old_meta['forecast_year']}년 · 당시 학습 종료: {old_meta['train_end']} · 저장 시각: {old_meta['created_at_utc']}")
                assessed = evaluate_snapshot(
                    old_meta, old_frames, result["a_monthly"], result["p_monthly"], cutoff,
                )
                for label, key in [("입항수", "arrivals"), ("승객수", "passengers")]:
                    frame, scores = assessed[key]
                    st.markdown(f"#### {label} · 당시 예측의 실제 실적 검증")
                    if scores["유효월"]:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAPE", fmt_number(scores["MAPE"], "%", 2))
                        m2.metric("WAPE", fmt_number(scores["WAPE"], "%", 2))
                        m3.metric("95% 근사 구간 포함률", fmt_number(scores["포함률"], "%", 1))
                        st.caption(f"실제값이 확정된 {scores['유효월']}개월만 평가. MAPE는 실제 0/결측 월 제외, WAPE는 유효월만 합산합니다.")
                        st.dataframe(frame.round(2), hide_index=True, use_container_width=True)
                        download_csv(f"{label} 실제 평가 CSV", frame,
                                     f"evaluation_{key}_{old_meta['forecast_year']}.csv", "eval_"+key)
                    else:
                        st.info(f"{label}: 이 기록을 만든 이후 실제 실적이 확정된 예측월이 아직 없습니다. 향후 원자료를 갱신하면 평가할 수 있습니다.")
            except Exception as exc:
                st.error(f"이전 예측 기록을 불러오거나 평가할 수 없습니다: {exc}")
        else:
            st.info("과거에 저장한 예측 JSON을 선택하면, 당시 예측값을 그대로 두고 이번에 추가된 실제 완료월만 평가합니다. 과거 예측을 현재 모델로 다시 학습하지 않습니다.")

with mtab:
    with st.container(key="page_workflow", border=False):
        st.markdown('<div class="section-kicker">07 / HOW IT WORKS</div>', unsafe_allow_html=True)
        st.subheader("자료 업로드 → 검증 → 재학습 → 선택 연도 전망")
        show_video()
        st.markdown("""<div class="stage">① 누적 원자료 업로드 → ② 실적 완료월 선택·확인 → ③ 무입항·승객 결측 검증 → ④ 두 ETS 모형 재학습 → ⑤ 목표 연도까지 끊김 없이 다이렉트 예측 → ⑥ 선택 연도의 12개월 추출·일정 비교 → ⑦ 예측 JSON 보관 → ⑧ 실적 확정 후 정확도 검증</div>""", unsafe_allow_html=True)
    with st.container(key="page_explainer", border=False):
        st.markdown('<div class="section-kicker">08 / MODEL & METRICS</div>', unsafe_allow_html=True)
        st.subheader("학습·예측 기준과 평가 지표")
        st.markdown(f"""
        - **입항수:** ETS(A,N,A), 추세 미적용 계절모형 · 학습 **2013.01~{cutoff}** · **다이렉트 예측만**
        - **승객수:** ETS(A,Ad,A), 감쇠 추세 계절모형 · 학습 **2023.01~{cutoff}**
        - **예측 기간:** {str(pd.Period(cutoff,freq='M')+1)}부터 {selected_year}.12까지 연속 계산, 화면에는 **{selected_year}년 1~12월**만 표시
        - **오차:** 실제−예측
        - **MAPE:** 월별 절대백분율오차 평균(실제 0 또는 결측은 제외)
        - **WAPE:** 절대오차 합계 ÷ 실제 합계 × 100(실제값이 유효한 월만)
        - **포함률:** 실제가 95% 근사 예측구간 안에 들어간 유효월 비율
        """)
        st.caption("예정 스케줄상 정원은 실제 탑승 승객수가 아닙니다. 예정 대비 편차로 MAPE·WAPE를 계산하지 않습니다.")
        st.warning("95% 구간은 고정된 모수 조건에서의 근사 구간입니다. 실제 포함률 95%를 보장하지 않으며, 장기 예측 시 일정·정책·선박 구성의 변화가 반영되지 않을 수 있습니다.")
        with st.expander("승객수 학습 자료 및 데이터 처리 내역"):
            st.dataframe(result["p_monthly"], hide_index=True, use_container_width=True)
            if result["absent_arrival_months"]:
                st.caption(f"입항수 기록 없는 월 {len(result['absent_arrival_months'])}개 중 최초 기준선 이후의 무기록 월은 확인 후 0으로 처리했습니다. 기준선 이전의 처리 방식은 기존 노트북을 유지합니다.")
        st.caption("원자료 CSV는 서버의 영구 저장소로 자동 저장하지 않습니다. 공개 클라우드에 배포한다면 파일 접근 권한과 플랫폼 데이터 정책을 별도로 확인하세요.")
