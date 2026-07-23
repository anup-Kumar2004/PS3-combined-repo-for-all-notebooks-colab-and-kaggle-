from pydantic import BaseModel
from typing import Optional

class SOHRequest(BaseModel):
    battery_id: str
    cycle_number: int
    duration: float
    v_min: float
    v_max: float
    v_mean: float
    v_std: float
    i_mean: float
    i_std: float
    t_max: float
    t_mean: float
    t_std: float
    v_drop_rate: float
    ambient_temperature: float
    duration_roll5: float
    duration_trend: float
    v_mean_roll5: float
    v_mean_trend: float
    t_mean_roll5: float
    t_mean_trend: float
    i_mean_roll5: float
    i_mean_trend: float
    SOH_prev: float
    SOH_change: float
    SOH_roll5: float

class SOHResponse(BaseModel):
    battery_id: str
    cycle_number: int
    soh_predicted: float
    health_status: str
    health_color: str

class RULResponse(BaseModel):
    battery_id: str
    cycle_number: int
    rul_cycles: Optional[int]
    days_light: Optional[float]
    days_average: Optional[float]
    days_heavy: Optional[float]

class BatteryHistoryResponse(BaseModel):
    battery_id: str
    cycles: list
    soh_actual: list
    soh_predicted: list
    rul_predicted: list
    rul_actual: list
    has_rul: bool
    dataset: str