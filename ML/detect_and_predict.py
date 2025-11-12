import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Charger le modèle
path = "ML/models/emotion_model.h5"
model = load_model(path)
print("✅ Model Loaded Successfully")

# Charger Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Erreur : Haar Cascade non trouvé.")
else:
    print("✅ Haar Cascade chargé avec succès")

# Labels des émotions
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Dossier contenant les images
folder_path = "images"  # ton dossier contenant les images

# Lister toutes les images du dossier
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Nombre d'images à tester : {len(image_files)}")

for image_name in image_files:
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"⚠️ Impossible de lire l'image : {image_name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    print(f"{image_name} → Nombre de visages détectés : {len(faces)}")
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=(0, -1))  # (1,48,48,1)
        
        prediction = model.predict(face_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100  # pourcentage
        
        label = labels[predicted_class] if predicted_class < len(labels) else f"Classe {predicted_class}"
        label_text = f"{label} ({confidence:.2f}%)"
        
        # Dessiner sur l'image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Afficher l'image avec la prédiction et la confiance
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Détection et Prédiction : {image_name}")
    plt.show()
