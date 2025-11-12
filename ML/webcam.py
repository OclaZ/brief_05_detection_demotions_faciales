import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque

# === Charger le modèle ===
path = "ML/models/emotion_model.h5"
model = load_model(path)
print("✅ Modèle chargé avec succès")

# === Charger Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Erreur : Haar Cascade non trouvé.")
else:
    print("✅ Haar Cascade chargé avec succès")

# === Labels des émotions ===
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# === Ouvrir la webcam ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise Exception("❌ Impossible d'accéder à la webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Webcam activée — appuyez sur 'q' pour quitter")

# === Variables pour FPS & stabilité ===
prev_time = 0
fps = 0
smooth_predictions = deque(maxlen=5)  # garder les 5 dernières prédictions pour lisser le résultat

def get_bar_color(conf):
    """Retourne une couleur selon la confiance"""
    if conf > 80:
        return (0, 255, 0)   # vert
    elif conf > 50:
        return (0, 255, 255) # jaune
    else:
        return (0, 0, 255)   # rouge

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Erreur de lecture du flux vidéo.")
        break

    # Réduire la taille du frame pour de meilleures performances
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Détecter les visages ===
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    emotion_display = "No face detected"
    conf_display = 0.0

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=(0, -1))  # (1,48,48,1)

        # === Prédire ===
        prediction = model.predict(face_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        smooth_predictions.append((predicted_class, confidence))
        # Moyenne lissée
        avg_class = int(np.mean([p[0] for p in smooth_predictions]))
        avg_conf = np.mean([p[1] for p in smooth_predictions])

        label = labels[avg_class] if avg_class < len(labels) else f"Classe {avg_class}"
        label_text = f"{label} ({avg_conf:.1f}%)"

        emotion_display = label
        conf_display = avg_conf

        # === Dessiner le rectangle ===
        color = get_bar_color(avg_conf)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # === Barre de confiance ===
        bar_width = int((avg_conf / 100) * w)
        cv2.rectangle(frame, (x, y + h + 10), (x + bar_width, y + h + 30), color, -1)
        cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 30), (255, 255, 255), 2)

    # === Calcul des FPS ===
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # === Overlay infos ===
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if faces is not None and len(faces) > 0:
        cv2.putText(frame, f"Emotion: {emotion_display}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Confiance: {conf_display:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # === Affichage ===
    cv2.imshow("🧠 Détection des émotions (Appuyez sur 'q' pour quitter)", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Libérer les ressources ===
cap.release()
cv2.destroyAllWindows()
print("👋 Fermé proprement.")
