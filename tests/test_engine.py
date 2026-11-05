import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest

from engine import (
    arrival_monthly, passenger_monthly, compare_schedule,
    read_csv_auto, load_arrival_rows, load_schedule,
    month_labels, missing_passenger_months, historical_scoring,
)


def sample_raw():
    rows = []
    for date in pd.date_range("2013-01-01", "2026-09-01", freq="MS"):
        if 2020 <= date.year <= 2022:
            continue
        if date.strftime("%Y-%m") in {"2023-01", "2023-02", "2023-12", "2024-01"}:
            continue
        amount = 2 + date.month % 3
        for i in range(amount):
            rows.append({"입출": "입항", "입항일시": date + pd.Timedelta(hours=i),
                         "호출부호": f"CALL{i}", "선명": f"SHIP{i}", "승객": 400 + 12 * date.month + i})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def test_original_aggregates():
    arrivals, info = load_arrival_rows(sample_raw())
    assert info["duplicate_count"] == 0
    a, absent = arrival_monthly(arrivals)
    assert len(a) == 165
    assert a["실제입항수"].sum() == len(arrivals)
    assert a.loc[a["연월"] == "2021-07", "실제입항수"].iloc[0] == 0
    p, log = passenger_monthly(arrivals, ["2023-01", "2023-02", "2023-12", "2024-01"])
    assert len(p) == 45
    assert p.loc[p["연월"] == "2023-02", "학습용승객수"].iloc[0] == 0
    assert p["학습용승객수"].notna().all()


def test_confirmed_zero_required():
    arrivals, _ = load_arrival_rows(sample_raw())
    with pytest.raises(ValueError, match="학습할 수 없는 월"):
        passenger_monthly(arrivals, [])


def test_schedule_and_comparison_missing_month():
    schedule = pd.DataFrame({"입항예정일": ["2027-01-03", "2027-01-11"], "선명": ["A", "B"],
                             "승객정원수": [100, 200], "예정승객수": [80, 160]})
    s = load_schedule(schedule.to_csv(index=False).encode("utf-8-sig"))
    months = month_labels("2027-01", "2027-12")
    p_forecast = pd.DataFrame({"연월": months, "예측승객수": np.repeat(200.0, 12)})
    a_forecast = pd.DataFrame({"연월": months, "예측입항수": np.repeat(3.0, 12)})
    p, a, info = compare_schedule(p_forecast, a_forecast, s)
    assert len(p) == 12 and len(a) == 12
    assert p.loc[0, "예정 승객수"] == 300
    assert p.loc[0, "차이"] == 100
    assert a.loc[0, "차이"] == 1
    assert pd.isna(p.loc[1, "예정 승객수"])
    assert info["observed_schedule_months"] == 1
    assert p["예정 승객수"].iloc[0] == s["승객정원수"].sum()
    assert "실제 예정 승객수" not in p.columns


def test_zero_denominator_is_unavailable():
    months = month_labels("2027-01", "2027-12")
    p_forecast = pd.DataFrame({"연월": months, "예측승객수": np.repeat(50.0, 12)})
    a_forecast = pd.DataFrame({"연월": months, "예측입항수": np.repeat(1.0, 12)})
    s = load_schedule(pd.DataFrame({"입항예정일": ["2027-01-01"], "선명": ["C"], "승객정원수": [0]}).to_csv(index=False).encode())
    p, a, _ = compare_schedule(p_forecast, a_forecast, s)
    assert pd.isna(p.loc[0, "예정대비절대차이율(%)"])
    assert a.loc[0, "예정건수대비절대차이율(%)"] == 0


def test_metrics_exclude_zero():
    v = historical_scoring(pd.Series([0., 100., 200.]), pd.Series([2., 110., 150.]))
    assert v["유효월"] == 3
    assert round(v["MAPE"], 5) == 17.5
    assert round(v["WAPE"], 5) == round(62 / 300 * 100, 5)


def test_model_fits_and_forecast_2027():
    # Synthetic fixture only: never presented as authentic port forecast.
    from engine import forecast_arrivals, forecast_passengers
    arrivals, _ = load_arrival_rows(sample_raw())
    m_a, _ = arrival_monthly(arrivals)
    m_p, _ = passenger_monthly(arrivals, ["2023-01", "2023-02", "2023-12", "2024-01"])
    a, a_info = forecast_arrivals(m_a)
    p, p_info = forecast_passengers(m_p)
    for frame in (a, p):
        assert frame["연월"].tolist() == month_labels("2027-01", "2027-12")
        assert frame["95% 하한"].le(frame["95% 상한"]).all()
    assert a_info["model"] == "ETS(A,N,A)"
    assert p_info["model"] == "ETS(A,Ad,A)"
