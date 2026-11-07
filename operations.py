"""예정 스케줄에 기반한 터미널 사전 운영 검토(규칙 기반 보조 지표).

이 모듈은 예측·운영 확정 시스템이 아닙니다. 입력 스케줄에서 날짜/시각과
선석 표기만 비교하며 실제 승객, 혼잡도, 선석 이용 가능 여부는 판단하지 않습니다.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperatingThresholds:
    arrivals_per_day: int = 2
    planned_capacity_per_day: int = 8000
    near_arrival_minutes: int = 90

    def __post_init__(self):
        if not 2 <= self.arrivals_per_day <= 20:
            raise ValueError("하루 입항 검토기준은 2~20건에서 설정해주세요.")
        if not 100 <= self.planned_capacity_per_day <= 100000:
            raise ValueError("하루 정원 검토기준은 100~100,000명에서 설정해주세요.")
        if not 15 <= self.near_arrival_minutes <= 360:
            raise ValueError("입항 시간 간격 검토기준은 15~360분에서 설정해주세요.")


def examine_schedule(schedule: pd.DataFrame | None, year: int,
                     thresholds: OperatingThresholds | None = None) -> dict:
    """Return daily review table and vessel detail for selected year.

    Thresholds are *user-set review triggers*, not port-authority regulations or
    formal resource-capacity limits. No unscheduled day is presumed zero unless
    complete schedule has been independently confirmed elsewhere.
    """
    thresholds = thresholds or OperatingThresholds()
    empty = {"daily": pd.DataFrame(), "vessels": pd.DataFrame(),
             "overlaps": pd.DataFrame(), "warnings": [], "days": 0, "events": 0}
    if schedule is None or schedule.empty:
        return empty
    needed = {"입항예정일", "선명", "승객정원수"}
    if not needed.issubset(schedule.columns):
        raise ValueError("운영 점검을 위한 일정 필수 열이 없습니다: " + ", ".join(sorted(needed - set(schedule.columns))))
    s = schedule.copy()
    s["입항예정일"] = pd.to_datetime(s["입항예정일"], errors="coerce")
    if s["입항예정일"].isna().any():
        raise ValueError("일부 입항예정일을 날짜로 변환할 수 없습니다.")
    s = s.loc[s["입항예정일"].dt.year.eq(int(year))].copy()
    if s.empty:
        return empty
    s = s.sort_values("입항예정일").reset_index(drop=True)
    s["일자"] = s["입항예정일"].dt.strftime("%Y-%m-%d")
    s["연월"] = s["입항예정일"].dt.strftime("%Y-%m")
    s["승객정원수"] = pd.to_numeric(s["승객정원수"], errors="coerce")
    if s["승객정원수"].isna().any() or (s["승객정원수"] < 0).any() or not np.isfinite(s["승객정원수"]).all():
        raise ValueError("승객정원수에 결측/음수/비정상 수치가 있습니다.")
    warnings=[]
    if "입항부두" not in s.columns:
        s["입항부두"] = pd.NA
        warnings.append("입항부두 열이 없어 동일 선석 시간 중첩은 검사하지 못했습니다.")
    s["입항부두"] = s["입항부두"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    if s["입항부두"].isna().any() or s["입항부두"].eq("").any():
        warnings.append("선석 정보가 없는 일정은 동일 선석 중첩 검사에서 제외됩니다.")
    if "출항예정일" in s.columns:
        s["출항예정일"] = pd.to_datetime(s["출항예정일"], errors="coerce")
        invalid_depart = s["출항예정일"].isna() | s["출항예정일"].lt(s["입항예정일"])
        if invalid_depart.any():
            warnings.append(f"출항예정일이 없거나 입항 전으로 기록된 {int(invalid_depart.sum())}건은 선석 중첩 계산에서 제외했습니다.")
        s.loc[invalid_depart,"출항예정일"] = pd.NaT
    else:
        s["출항예정일"] = pd.NaT
        warnings.append("출항예정일 열이 없어 동일 선석 시간 중첩은 검사하지 못했습니다.")

    # Same berth with intersecting [arrival, departure) intervals.
    # A reported overlap requests human verification; berth labels need not
    # represent a physically exclusive berth in real operations.
    overlap_records = []
    counts_overlap: dict[str,int] = {}
    valid = s.loc[s["출항예정일"].notna() & s["입항부두"].notna() & s["입항부두"].ne("")].copy()
    for berth, block in valid.groupby("입항부두", sort=True):
        active=[]  # (row index, departure timestamp)
        for idx, cur in block.sort_values("입항예정일").iterrows():
            arrived=cur["입항예정일"]
            active=[(prev,departure) for prev,departure in active if departure > arrived]
            for prev,_ in active:
                prior=s.loc[prev]
                overlap_records.append({
                    "선석":str(berth), "선박1":str(prior["선명"]),
                    "입항1":str(prior["입항예정일"]),"출항1":str(prior["출항예정일"]),
                    "선박2":str(cur["선명"]), "입항2":str(cur["입항예정일"]),
                    "출항2":str(cur["출항예정일"]),
                })
                for date in {str(prior["일자"]),str(cur["일자"])}:
                    counts_overlap[date]=counts_overlap.get(date,0)+1
            active.append((idx,cur["출항예정일"]))
    overlaps=pd.DataFrame(overlap_records,columns=["선석","선박1","입항1","출항1","선박2","입항2","출항2"])

    daily_rows=[]
    for day, block in s.groupby("일자", sort=True):
        block=block.sort_values("입항예정일")
        n=len(block)
        total=float(block["승객정원수"].sum())
        morning=int((block["입항예정일"].dt.hour<12).sum())
        near=0
        if n>=2:
            gaps=block["입항예정일"].diff().dropna().dt.total_seconds()/60
            near=int(gaps.le(thresholds.near_arrival_minutes).sum())
        overlap_count=counts_overlap.get(day,0)
        reasons=[]
        if n >= thresholds.arrivals_per_day:
            reasons.append(f"동일 날짜 {n}척 입항")
        if total >= thresholds.planned_capacity_per_day:
            reasons.append(f"예정 정원 {total:,.0f}명")
        if near:
            reasons.append(f"{thresholds.near_arrival_minutes}분 이내 인접 입항 {near}쌍")
        if overlap_count:
            reasons.append(f"동일 선석 시간 중첩 확인 {overlap_count}쌍")
        daily_rows.append({
            "일자":day, "연월":block["연월"].iloc[0],"예정 입항수":n,
            "예정 승객수":total,"오전 입항수":morning,
            "근접 입항쌍":near,"동일 선석 중첩쌍":overlap_count,
            "점검 항목수":len(reasons),
            "운영 점검 사유":" · ".join(reasons) if reasons else "설정 기준 해당 없음",
            "확인 상태":"미확인", "담당자":"", "운영 메모":"",
        })
    daily=pd.DataFrame(daily_rows)
    display_cols=["일자", "입항예정일", "출항예정일", "입항부두", "선명", "승객정원수"]
    s["출항예정일"] = s["출항예정일"].dt.strftime("%Y-%m-%d %H:%M")
    s["입항예정일"] = s["입항예정일"].dt.strftime("%Y-%m-%d %H:%M")
    vessels=s[display_cols].rename(columns={"승객정원수":"예정 승객수(정원 기반)"})
    return {"daily": daily, "vessels": vessels, "overlaps":overlaps,
            "warnings": warnings, "days":len(daily), "events":len(s)}


def overview_metrics(result: dict) -> dict[str,int | float]:
    d=result["daily"]
    if d.empty:
        return {"scheduled_calls":0,"review_days":0,"overlap_pairs":0,"max_day_capacity":0.0,"max_day_calls":0}
    return {"scheduled_calls":int(d["예정 입항수"].sum()),
            "review_days":int((d["점검 항목수"]>0).sum()),
            "overlap_pairs":int(len(result["overlaps"])),
            "max_day_capacity":float(d["예정 승객수"].max()),
            "max_day_calls":int(d["예정 입항수"].max())}
