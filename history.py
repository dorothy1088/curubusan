"""Portable, read-only snapshots for ETS forecasts and later actual-outcome evaluation."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from engine import historical_scoring, month_labels

SCHEMA_VERSION = 2
FORECAST_COLUMNS = {
    "arrivals": ["연월", "예측입항수", "95% 하한", "95% 상한"],
    "passengers": ["연월", "예측승객수", "95% 하한", "95% 상한"],
}


def create_snapshot(cutoff: str, year: int, a_forecast: pd.DataFrame, p_forecast: pd.DataFrame,
                    a_meta: dict, p_meta: dict, raw_content: bytes) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_id": uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end": cutoff,
        "forecast_year": int(year),
        "raw_sha256": hashlib.sha256(raw_content).hexdigest(),
        "models": {
            "arrivals": {"name": a_meta["model"], "train_start": a_meta["train_start"],
                         "train_end": a_meta["train_end"], "seed": a_meta["seed"],
                         "simulation_paths": a_meta["simulations"], "method": "direct"},
            "passengers": {"name": p_meta["model"], "train_start": p_meta["train_start"],
                           "train_end": p_meta["train_end"], "seed": p_meta["seed"],
                           "simulation_paths": p_meta["simulations"]},
        },
        "forecast": {
            "arrivals": a_forecast[FORECAST_COLUMNS["arrivals"]].to_dict("records"),
            "passengers": p_forecast[FORECAST_COLUMNS["passengers"]].to_dict("records"),
        },
    }
    # A snapshot has no raw input rows. JSON must not contain NaN/Infinity tokens.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def snapshot_bytes(snapshot: dict) -> bytes:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def parse_snapshot(content: bytes) -> tuple[dict, dict[str, pd.DataFrame]]:
    if len(content) > 5_000_000:
        raise ValueError("예측 기록 JSON이 너무 큽니다(최대 5MB).")
    try:
        obj = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("예측 기록 JSON 형식을 확인해주세요.") from exc
    if not isinstance(obj, dict) or obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("지원하는 예측 기록 JSON이 아닙니다(형식 버전 2).")
    year = int(obj["forecast_year"])
    cutoff = str(pd.Period(obj["train_end"], freq="M"))
    if not 2027 <= year <= 2200 or year <= int(cutoff[:4]):
        raise ValueError("예측 연도와 학습 종료월이 잘못되었습니다.")
    models = obj.get("models")
    if not isinstance(models, dict) or models.get("arrivals", {}).get("name") != "ETS(A,N,A)" or models.get("passengers", {}).get("name") != "ETS(A,Ad,A)":
        raise ValueError("예측 기록의 모델 정보가 올바르지 않습니다.")
    frames = {}
    for key, columns in FORECAST_COLUMNS.items():
        frame = pd.DataFrame(obj["forecast"][key])
        if frame.empty or not set(columns).issubset(frame.columns):
            raise ValueError(f"{key} 예측 기록에 필요한 열이 없습니다.")
        if frame["연월"].tolist() != month_labels(f"{year}-01", f"{year}-12"):
            raise ValueError("예측 기록의 월 순서/범위가 12개월과 일치하지 않습니다.")
        for col in columns[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[columns[1:]].to_numpy(dtype=float)).all():
            raise ValueError("예측 기록에 비정상 수치가 있습니다.")
        if (frame[columns[1:]] < 0).any().any() or (frame["95% 하한"] > frame["95% 상한"]).any():
            raise ValueError("예측 기록의 구간/예측값이 유효하지 않습니다.")
        frames[key] = frame[columns].copy()
    return obj, frames


def evaluate_snapshot(snapshot: dict, forecast_frames: dict[str, pd.DataFrame],
                      arrivals_monthly: pd.DataFrame, passengers_monthly: pd.DataFrame,
                      current_cutoff: str) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Evaluate only months after the saved cutoff and by the newly confirmed cutoff."""
    saved_cutoff = snapshot["train_end"]
    results = {}
    for key, actual_frame, actual_col, pred_col in [
        ("arrivals", arrivals_monthly, "실제입항수", "예측입항수"),
        ("passengers", passengers_monthly, "실제승객수", "예측승객수"),
    ]:
        df = forecast_frames[key].merge(
            actual_frame[["연월", actual_col]], on="연월", how="left", validate="one_to_one"
        )
        period_valid = df["연월"].gt(saved_cutoff) & df["연월"].le(current_cutoff)
        df["실제 실적"] = df[actual_col].where(period_valid)
        df["오차(실제-예측)"] = df["실제 실적"] - df[pred_col]
        df["95% 구간 포함"] = pd.Series(pd.NA, dtype="boolean", index=df.index)
        ok = df["실제 실적"].notna() & df["95% 하한"].notna() & df["95% 상한"].notna()
        df.loc[ok, "95% 구간 포함"] = df.loc[ok, "실제 실적"].between(df.loc[ok, "95% 하한"], df.loc[ok, "95% 상한"])
        score = historical_scoring(df["실제 실적"], df[pred_col], df["95% 하한"], df["95% 상한"])
        out = df[["연월", "실제 실적", pred_col, "오차(실제-예측)", "95% 하한", "95% 상한", "95% 구간 포함"]]
        results[key] = (out, score)
    return results
