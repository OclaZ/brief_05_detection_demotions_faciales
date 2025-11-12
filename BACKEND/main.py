from fastapi import FastAPI
from routes.predict import router as predict_router
from core.database import Base, engine
from models import predictions

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Emotion Detection API")

# Include routes
app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Welcome to Emotion Detection API 👋"}
