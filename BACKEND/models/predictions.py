from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from BACKEND.core.database import Base

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, emotion={self.emotion}, confidence={self.confidence})>"