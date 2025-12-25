from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LogEntry(BaseModel):
    message: str
    component: Optional[str] = ""
    level: Optional[str] = ""


class LogSequenceRequest(BaseModel):
    logs: list[LogEntry] = Field(..., min_length=1, description="Список лог-записей")


class AnomalyResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float
    num_events: int


class HistoryItem(BaseModel):
    id: int
    request_type: str
    processing_time: float
    input_data_size: Optional[int]
    status_code: int
    result: Optional[str]
    error_message: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class StatsResponse(BaseModel):
    total_requests: int
    mean_processing_time: float
    median_processing_time: float
    percentile_95_processing_time: float
    percentile_99_processing_time: float
    average_input_size: Optional[float]


class UserCreate(BaseModel):
    username: str
    password: str = Field(..., min_length=1, max_length=72)
    is_admin: bool = False

    @field_validator('password')
    @classmethod
    def validate_password_bytes(cls, v: str) -> str:
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password cannot be longer than 72 bytes')
        return v


class UserResponse(BaseModel):
    id: int
    username: str
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
