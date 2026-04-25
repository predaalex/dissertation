from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime


class EmotionState(BaseModel):
    emotion_label: str
    confidence: float | None = None
    updated_at: datetime


class EmotionPredictionResponse(BaseModel):
    emotion_label: str
    emotion_index: int
    confidence: float
    probabilities: dict[str, float]


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class SessionStateResponse(BaseModel):
    session_id: str
    created_at: datetime
    latest_emotion: EmotionState | None = None
    history: list[SessionMessage]


class HealthResponse(BaseModel):
    status: str
    services: dict[str, bool]
    model: str


class EmotionUpdateRequest(BaseModel):
    emotion_label: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TranscriptionResponse(BaseModel):
    transcript: str
    backend: str


class OllamaStreamEvent(BaseModel):
    response: str = ""
    done: bool = False
    total_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    done_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
