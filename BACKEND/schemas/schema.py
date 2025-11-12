from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    message: str
    
    class Config:
        from_attributes = True

class PredictionDB(BaseModel):
    id: int
    emotion: str
    confidence: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class HistoryResponse(BaseModel):
    count: int
    predictions: List[PredictionDB]