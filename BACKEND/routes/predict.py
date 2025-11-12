from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import cv2
import io
from PIL import Image
from tensorflow.keras.models import load_model

from core.database import get_db
from models.predictions import Prediction
from schemas.schema import PredictionResponse,HistoryResponse

router = APIRouter(prefix="/api", tags=["Prediction"])

# Load CNN model once
model = load_model("emotion_model.h5")
print("✅ CNN model loaded successfully")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Erreur : Haar Cascade non trouvé.")
else:
    print("✅ Haar Cascade chargé avec succès")

# Emotion labels (adjust according to your model)
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


@router.post("/predict_emotion", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Read the uploaded file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert to OpenCV format
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="Aucun visage détecté.")

        # Take first face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Preprocess for CNN
        face_resized = cv2.resize(face_roi, (48, 48))
        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=(0, -1))

        # Predict
        prediction = model.predict(face_input)
        predicted_index = int(np.argmax(prediction))
        predicted_label = labels[predicted_index]
        confidence = float(np.max(prediction))

        # Save to database
        record = Prediction(emotion=predicted_label, confidence=confidence)
        db.add(record)
        db.commit()
        db.refresh(record)

        # Return response
        return JSONResponse(content={
            "emotion": predicted_label,
            "confidence": round(confidence, 3),
            "message": "Prediction successful ✅"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=HistoryResponse)
async def get_history(db: Session = Depends(get_db)):
    """
    Récupère l'historique complet des prédictions d'émotions
    depuis la base de données PostgreSQL.
    """
    try:
        # Get all predictions ordered by most recent first
        predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
        
        # Count total predictions
        count = len(predictions)
        
        # Return response
        return {
            "count": count,
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de l'historique: {str(e)}")
