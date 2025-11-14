from fastapi import FastAPI
from BACKEND.routes.predict import router as predict_router
from BACKEND.core.database import Base, engine
from BACKEND.models import predictions

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Emotion Detection API")

# Include routes
app.include_router(predict_router)

@app.get("/")
def home():
    return {"message": "Welcome to Emotion Detection API 👋"}
