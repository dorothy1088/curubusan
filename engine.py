"""부산항 크루즈 ETS 월별 갱신형 예측 엔진.

원본 코드 기반:
* 입항수: forecast.ipynb — ETS(A,N,A), direct 15개월, residual bootstrap 1,000경로
* 승객수: 부산항_ETS_전체분석 — ETS(A,Ad,A), heuristic, scale=1000, 10,000경로

학습 시작월은 고정하고 사용자가 확인한 실적 완료월까지 학습기간을 확장한다.
선택 연도의 12개월만 추출하며, 입항수는 한 번 적합한 ETS 모형의 다이렉트 예측만 사용한다.
"""
from __future__ import annotations

from io import BytesIO
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

ARRIVAL_START = "2013-01"
PASSENGER_START = "2023-01"
CUTOFF = "2026-09"  # 최초 기준선 완료월; 재학습 시 cutoff 인자로 갱신
FORECAST_START = "2027-01"
FORECAST_END = "2027-12"
SEED_ARRIVALS = 20261029
SEED_PASSENGERS = 20261030
N_ARRIVALS = 1000
N_PASSENGERS = 10000
KNOWN_ZERO = ("2023-01", "2023-02", "2023-12", "2024-01")


def month_labels(start: str, end: str) -> list[str]:
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def read_csv_auto(content: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(BytesIO(content), encoding=encoding)
            df.columns = df.columns.astype(str).str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8-SIG 또는 CP949 CSV로 저장해주세요.")


def load_arrival_rows(content: bytes, remove_duplicates: bool = False, cutoff: str = CUTOFF) -> tuple[pd.DataFrame, dict]:
    """전체 누적 CSV에서 사용자가 확정한 cutoff 이하의 입항 기록만 반환."""
    data = read_csv_auto(content)
    required = {"입출", "입항일시", "호출부호", "선명", "승객"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("원자료의 필수 열이 없습니다: " + ", ".join(sorted(missing)))
    duplicates = int(data.duplicated().sum())
    if duplicates and not remove_duplicates:
        raise ValueError(f"원자료에 완전히 동일한 행 {duplicates}개가 있습니다. 중복 여부 확인 후 중복 제거에 동의해주세요.")
    if remove_duplicates:
        data = data.drop_duplicates().copy()
    arrivals = data.loc[data["입출"].astype("string").str.strip().eq("입항")].copy()
    if arrivals.empty:
        raise ValueError("입출 열에서 '입항' 행을 찾지 못했습니다.")
    arrivals["입항일시"] = pd.to_datetime(arrivals["입항일시"], errors="coerce")
    if arrivals["입항일시"].isna().any():
        raise ValueError("입항일시 날짜를 해석할 수 없는 행이 있습니다.")
    raw = arrivals["승객"].astype("string").str.replace(",", "", regex=False).str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & ~raw.isin(["", "-", "nan", "None", "NaN"]) & num.isna()
    if invalid.any():
        raise ValueError(f"승객 열에서 숫자가 아닌 값 {int(invalid.sum())}건을 확인해주세요.")
    if (num.dropna() < 0).any() or not np.isfinite(num.dropna()).all():
        raise ValueError("승객 열에 음수 또는 무한대가 있습니다.")
    arrivals["승객"] = num.astype(float)
    arrivals["호출부호"] = arrivals["호출부호"].astype("string").str.strip()
    arrivals["연월"] = arrivals["입항일시"].dt.strftime("%Y-%m")
    arrivals = arrivals.sort_values("입항일시").reset_index(drop=True)
    if pd.Period(cutoff, freq="M") < pd.Period(CUTOFF, freq="M"):
        raise ValueError("최초 학습 완료월 2026-09 이전으로 되돌릴 수 없습니다.")
    in_range = arrivals["연월"].between(ARRIVAL_START, cutoff)
    meta = {
        "duplicate_count": duplicates,
        "ignored_after_cutoff": int((arrivals["연월"] > cutoff).sum()),
        "ignored_before_start": int((arrivals["연월"] < ARRIVAL_START).sum()),
        "raw_arrival_rows": int(len(arrivals)),
        "last_recorded_month": str(arrivals["연월"].max()),
        "cutoff": cutoff,
    }
    if not in_range.any():
        raise ValueError("선택된 완료월까지 입항 기록이 없습니다.")
    return arrivals.loc[in_range].copy().reset_index(drop=True), meta


def missing_passenger_months(rows: pd.DataFrame, cutoff: str = CUTOFF) -> list[str]:
    observed = set(rows.loc[rows["연월"].between(PASSENGER_START, cutoff), "연월"].tolist())
    return [month for month in month_labels(PASSENGER_START, cutoff) if month not in observed]


def new_arrival_absences(rows: pd.DataFrame, cutoff: str = CUTOFF) -> list[str]:
    """최초 기준선 이후의 무기록 월: 임의 0건 처리 금지."""
    if cutoff <= CUTOFF:
        return []
    observed = set(rows["연월"].tolist())
    return [m for m in month_labels(str(pd.Period(CUTOFF, freq="M") + 1), cutoff) if m not in observed]


def arrival_monthly(rows: pd.DataFrame, cutoff: str = CUTOFF,
                    confirmed_recent_zeros: Iterable[str] = ()) -> tuple[pd.DataFrame, list[str]]:
    months = month_labels(ARRIVAL_START, cutoff)
    recent = rows.loc[rows["연월"].between(ARRIVAL_START, cutoff)].copy()
    absent_new = new_arrival_absences(recent, cutoff)
    unapproved = [m for m in absent_new if m not in set(confirmed_recent_zeros)]
    if unapproved:
        raise ValueError("새 실적 기간의 무기록 월은 0건인지 확인이 필요합니다: " + ", ".join(unapproved))
    # 최초 2013-2026.09 무기록 월은 기존 forecast.ipynb 집계 관례(0건)를 보존한다.
    counts = recent.groupby("연월").size().reindex(months, fill_value=0)
    result = pd.DataFrame({"연월": months, "실제입항수": counts.to_numpy(dtype=float)})
    absent = [m for m in months if m not in set(recent["연월"])]
    if int(result["실제입항수"].sum()) != len(recent):
        raise ValueError("입항 건수 재집계 검증에 실패했습니다.")
    return result, absent


def passenger_monthly(rows: pd.DataFrame, confirmed_zeros: Iterable[str], cutoff: str = CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = month_labels(PASSENGER_START, cutoff)
    frame = rows.loc[rows["연월"].between(PASSENGER_START, cutoff)].copy()
    if frame.empty:
        raise ValueError("2023년 이후의 입항 기록이 없습니다.")
    agg = frame.groupby("연월").agg(
        입항건수=("승객", "size"),
        유효기록수=("승객", "count"),
        확인승객합계=("승객", lambda x: x.sum(min_count=1)),
    ).reindex(periods)
    absent = agg["입항건수"].isna()
    approved = set(confirmed_zeros)
    allowed = set(periods)
    if not approved.issubset(allowed):
        raise ValueError("확인 무입항 월은 승객 학습기간 내에서만 지정할 수 있습니다.")
    zero = absent & agg.index.isin(approved)
    agg.loc[absent, ["입항건수", "유효기록수"]] = 0
    agg.loc[zero, "확인승객합계"] = 0
    agg["결측입항건수"] = agg["입항건수"] - agg["유효기록수"]
    agg["실제승객수"] = agg["확인승객합계"].where(agg["결측입항건수"].eq(0))
    agg["자료상태"] = np.where(
        absent,
        np.where(zero, "확인된 무입항 0명", "미확인 무기록 월"),
        np.where(agg["결측입항건수"].gt(0), "승객 기록 결측", "승객 기록 완전"),
    )
    frame["학습승객"] = frame["승객"].copy()
    logs: list[dict] = []
    # 이전 시점의 동일 호출부호 관측만 사용. 이미 보완한 값, 미래 자료는 참조 금지.
    for idx, row in frame.loc[frame["승객"].isna()].iterrows():
        history = frame.loc[
            (frame["입항일시"] < row["입항일시"])
            & frame["승객"].notna()
            & frame["호출부호"].eq(row["호출부호"])
        ]
        imputed = float(history["승객"].median()) if not history.empty else np.nan
        frame.loc[idx, "학습승객"] = imputed
        logs.append({
            "입항일시": str(row["입항일시"]),
            "선명": str(row["선명"]),
            "이전 관측 개수": int(len(history)),
            "보완 승객수": imputed,
        })
    training = frame.groupby("연월")["학습승객"].agg(lambda x: x.sum() if x.notna().all() else np.nan).reindex(periods)
    training.loc[zero] = 0
    agg["학습용승객수"] = training
    agg.index.name = "연월"
    out = agg.reset_index()
    bad_months = out.loc[out["학습용승객수"].isna(), "연월"].tolist()
    if bad_months:
        raise ValueError(
            "학습할 수 없는 월이 있습니다(미확인 무기록 또는 보완 불가 승객 결측): "
            + ", ".join(bad_months)
            + ". 무기록 월이 정말 0건인지 확인하거나 원자료를 보완해주세요."
        )
    if len(out) != len(periods):
        raise ValueError("승객수 학습기간의 월 수가 올바르지 않습니다.")
    return out, pd.DataFrame(logs, columns=["입항일시", "선명", "이전 관측 개수", "보완 승객수"])


def _as_array(data: pd.Series) -> np.ndarray:
    vals = np.asarray(data, dtype=float)
    if len(vals) < 24 or not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("ETS 학습에는 최소 24개월 이상의 연속된 비음수 관측값이 필요합니다.")
    return vals


def _check_horizon(cutoff: str, target_year: int) -> tuple[int, int, list[str]]:
    """Return h, 0-based index for first target month, target months."""
    end = pd.Period(f"{target_year}-12", freq="M")
    start = pd.Period(cutoff, freq="M")
    if target_year <= start.year:
        raise ValueError("예측 연도는 최종 실적 완료연도보다 이후여야 합니다. 과거 평가는 예측 기록 메뉴에서 확인하세요.")
    h = int(end.ordinal - start.ordinal)
    if h > 144:
        raise ValueError("최장 12년 이내로 예측 연도를 선택해주세요. 장기 전망은 근사 오차가 커집니다.")
    first_index = int(pd.Period(f"{target_year}-01", freq="M").ordinal - start.ordinal - 1)
    return h, first_index, month_labels(f"{target_year}-01", f"{target_year}-12")


def _future_year(forecast: np.ndarray, paths: np.ndarray, label: str,
                 cutoff: str, target_year: int) -> tuple[pd.DataFrame, tuple[float, float]]:
    h, index, year = _check_horizon(cutoff, target_year)
    if forecast.shape != (h,) or paths.shape[0] != h:
        raise ValueError("예측 시계열의 월 수가 일치하지 않습니다.")
    point = np.maximum(np.asarray(forecast, dtype=float), 0)[index:index+12]
    clipped = np.maximum(np.asarray(paths, dtype=float), 0)[index:index+12, :]
    if len(point) != 12 or clipped.shape[0] != 12:
        raise ValueError("선택 연도의 12개월 예측을 만들지 못했습니다.")
    lo, hi = np.quantile(clipped, [0.025, 0.975], axis=1)
    result = pd.DataFrame({"연월": year, f"예측{label}": point, "95% 하한": lo, "95% 상한": hi})
    if not np.isfinite(result.select_dtypes("number")).all().all():
        raise ValueError("예측 결과에 비정상적인 수치가 있습니다.")
    annual_interval = tuple(float(v) for v in np.quantile(clipped.sum(axis=0), [0.025, 0.975]))
    return result, annual_interval


def forecast_arrivals(monthly: pd.DataFrame, cutoff: str = CUTOFF, target_year: int = 2027) -> tuple[pd.DataFrame, dict]:
    y = _as_array(monthly["실제입항수"])
    if len(y) != len(month_labels(ARRIVAL_START, cutoff)):
        raise ValueError("입항수 월별 학습기간이 완전하지 않습니다.")
    h, _, _ = _check_horizon(cutoff, target_year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 원본 forecast.ipynb의 ETS(A,N,A) 학습·다이렉트 예측 유지.
        fit = ETSModel(y, error="add", trend=None, seasonal="add", seasonal_periods=12).fit(disp=False, maxiter=400)
    if not np.isfinite(fit.params).all():
        raise RuntimeError("입항수 ETS(A,N,A) 모수 계산에 실패했습니다.")
    fc = np.asarray(fit.forecast(h), dtype=float)
    residuals = np.asarray(fit.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("입항수 ETS 잔차가 없어 예측구간을 만들 수 없습니다.")
    bootstrap = np.random.default_rng(SEED_ARRIVALS).choice(residuals, size=(h, N_ARRIVALS), replace=True)
    sim = np.asarray(fit.simulate(
        nsimulations=h, anchor="end", repetitions=N_ARRIVALS, random_errors=bootstrap
    )).reshape(h, N_ARRIVALS)
    result, annual_interval = _future_year(fc, sim, "입항수", cutoff, target_year)
    meta = {
        "model": "ETS(A,N,A)", "train_start": ARRIVAL_START,
        "train_end": cutoff, "periods": len(y), "forecast_horizon_months": h,
        "forecast_year": target_year, "simulations": N_ARRIVALS, "seed": SEED_ARRIVALS,
        "annual_interval": annual_interval, "method": "direct",
    }
    return result, meta


def forecast_passengers(monthly: pd.DataFrame, cutoff: str = CUTOFF, target_year: int = 2027) -> tuple[pd.DataFrame, dict]:
    y = _as_array(monthly["학습용승객수"]) / 1000.0
    if len(y) != len(month_labels(PASSENGER_START, cutoff)):
        raise ValueError("승객수 월별 학습기간이 완전하지 않습니다.")
    h, _, _ = _check_horizon(cutoff, target_year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ETSModel(
            y, error="add", trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=12, initialization_method="heuristic",
        ).fit(disp=False, maxiter=2000)
    if not bool(fit.mle_retvals.get("converged", False)):
        raise RuntimeError("승객수 ETS(A,Ad,A) 모형이 수렴하지 않았습니다. 데이터 품질을 확인해주세요.")
    fc = np.maximum(np.asarray(fit.forecast(h), dtype=float), 0) * 1000
    # v0.14.6은 random_state를 사용하므로 기존 계산과 동일한 난수 흐름 유지.
    try:
        sim_values = fit.simulate(h, anchor="end", repetitions=N_PASSENGERS,
                                  rng=np.random.default_rng(SEED_PASSENGERS))
    except TypeError as exc:
        if "rng" not in str(exc):
            raise
        sim_values = fit.simulate(h, anchor="end", repetitions=N_PASSENGERS,
                                  random_state=np.random.RandomState(SEED_PASSENGERS))
    sim = np.asarray(sim_values, dtype=float).reshape(h, N_PASSENGERS) * 1000
    result, annual_interval = _future_year(fc, sim, "승객수", cutoff, target_year)
    meta = {
        "model": "ETS(A,Ad,A)", "train_start": PASSENGER_START,
        "train_end": cutoff, "periods": len(y), "forecast_horizon_months": h,
        "forecast_year": target_year, "simulations": N_PASSENGERS, "seed": SEED_PASSENGERS,
        "annual_interval": annual_interval,
    }
    return result, meta


def load_schedule(content: bytes) -> pd.DataFrame:
    schedule = read_csv_auto(content)
    required = {"입항예정일", "선명", "승객정원수"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError("예정 스케줄의 필수 열이 없습니다: " + ", ".join(sorted(missing)))
    exact_dups = int(schedule.duplicated().sum())
    if exact_dups:
        raise ValueError(f"예정 스케줄에 동일 행 {exact_dups}개가 있습니다. 중복 제거 후 업로드해주세요.")
    if schedule.empty:
        raise ValueError("예정 스케줄이 비어 있습니다.")
    dates = pd.to_datetime(schedule["입항예정일"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"예정 스케줄의 입항예정일 변환 실패: {int(dates.isna().sum())}개")
    schedule["입항예정일"] = dates
    schedule["선명"] = schedule["선명"].astype("string").str.strip()
    if schedule["선명"].isna().any() or schedule["선명"].eq("").any():
        raise ValueError("예정 스케줄에 비어 있는 선명이 있습니다.")
    if schedule.duplicated(["입항예정일", "선명"]).any():
        raise ValueError("동일 선명·입항예정일이 중복됐습니다. 서로 다른 회차라면 날짜/시각을 구분해주세요.")
    for c in ("승객정원수", "예정승객수"):
        if c not in schedule.columns:
            continue
        values = pd.to_numeric(schedule[c].astype("string").str.replace(",", "", regex=False), errors="coerce")
        if c == "승객정원수" and values.isna().any():
            raise ValueError("예정 스케줄의 승객정원수에 숫자가 아니거나 빈 값이 있습니다.")
        if (values.dropna() < 0).any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f"예정 스케줄의 {c} 값은 유효한 비음수여야 합니다.")
        schedule[c] = values.astype(float)
    schedule["연월"] = schedule["입항예정일"].dt.strftime("%Y-%m")
    return schedule


def compare_schedule(
    passenger_forecast: pd.DataFrame,
    arrival_forecast: pd.DataFrame,
    schedule: pd.DataFrame | None,
    complete_schedule: bool = False,
    target_year: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use uploaded schedule capacity as the display-only **예정 승객수** proxy.

    It is not observed or confirmed boarding. Do not confuse this capacity-derived
    measure with an independently observed '예정승객수' column or actual passenger counts.
    """
    months = pd.DataFrame({"연월": month_labels(f"{target_year}-01", f"{target_year}-12")})
    if schedule is None:
        cap = pd.DataFrame(columns=["연월", "예정입항건수합계", "승객정원합계"])
        outside = []
    else:
        outside = sorted(set(schedule["연월"]) - set(months["연월"]))
        records = schedule.loc[schedule["연월"].isin(months["연월"])].copy()
        cap = records.groupby("연월").agg(
            예정입항건수합계=("선명", "size"),
            승객정원합계=("승객정원수", "sum"),
        ).reset_index()
    base = months.merge(cap, on="연월", how="left", validate="one_to_one")
    if complete_schedule and schedule is not None:
        # Only explicitly verified full-year schedules permit interpreting an absent month as zero.
        base[["예정입항건수합계", "승객정원합계"]] = base[
            ["예정입항건수합계", "승객정원합계"]
        ].fillna(0.0)

    p = base.merge(passenger_forecast[["연월", "예측승객수"]], on="연월", validate="one_to_one")
    # UI label requested by user; source is still 승객정원수, NOT actual/confirmed passengers.
    p["예정 승객수"] = p["승객정원합계"]
    p["차이"] = p["예정 승객수"] - p["예측승객수"]
    p["예정대비절대차이율(%)"] = p["차이"].abs() / p["예정 승객수"].where(p["예정 승객수"] > 0) * 100
    p = p[["연월", "예정 승객수", "예측승객수", "차이", "예정대비절대차이율(%)"]]

    a = base.merge(arrival_forecast[["연월", "예측입항수"]], on="연월", validate="one_to_one")
    a["실제 예정 입항수"] = a["예정입항건수합계"]
    # Keep the original arrival notebook's signed difference: prediction minus scheduled.
    a["차이"] = a["예측입항수"] - a["예정입항건수합계"]
    a["예정건수대비절대차이율(%)"] = a["차이"].abs() / a["예정입항건수합계"].where(a["예정입항건수합계"] > 0) * 100
    a = a[["연월", "실제 예정 입항수", "예측입항수", "예정입항건수합계", "차이", "예정건수대비절대차이율(%)"]]
    info = {
        "outside_period": outside,
        "observed_schedule_months": int(base["예정입항건수합계"].notna().sum()),
        "complete_schedule": complete_schedule,
    }
    return p, a, info


def historical_scoring(actual: pd.Series, predicted: pd.Series, lower: pd.Series | None = None, upper: pd.Series | None = None) -> dict:
    """실제 완료된 월만 평가. 2027년 예정 정원을 정답으로 넣지 않는다."""
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna() & np.isfinite(a) & np.isfinite(p) & (a >= 0)
    if not valid.any():
        return {"오차": pd.Series(dtype=float), "MAPE": np.nan, "WAPE": np.nan, "포함률": np.nan, "유효월": 0}
    errors = a[valid] - p[valid]
    abs_errors = errors.abs()
    positive = a[valid] > 0
    mape = float((abs_errors[positive] / a[valid][positive] * 100).mean()) if positive.any() else np.nan
    wape = float(abs_errors.sum() / a[valid].sum() * 100) if a[valid].sum() > 0 else np.nan
    coverage = np.nan
    if lower is not None and upper is not None:
        l = pd.to_numeric(lower, errors="coerce")
        u = pd.to_numeric(upper, errors="coerce")
        mask = valid & l.notna() & u.notna()
        if mask.any():
            coverage = float(((a[mask] >= l[mask]) & (a[mask] <= u[mask])).mean() * 100)
    return {"오차": errors, "MAPE": mape, "WAPE": wape, "포함률": coverage, "유효월": int(valid.sum())}
