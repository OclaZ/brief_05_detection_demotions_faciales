from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    emotion: str
    confidence: float
    message: str

class PredictionDB(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    emotion: str
    confidence: float
    created_at: datetime

class HistoryResponse(BaseModel):
    count: int
    predictions: List[PredictionDB]