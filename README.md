<div align="center">
  <br />
  <img src="https://www.simplon.ma/images/Simplon_Maghreb_Rouge.png" alt="Simplon Maghreb Logo" width="300"/>
  <br /><br />

  <div>
    <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logo=python&logoColor=white&color=3776AB" />
    <img src="https://img.shields.io/badge/-TensorFlow-black?style=for-the-badge&logo=tensorflow&logoColor=white&color=FF6F00" />
    <img src="https://img.shields.io/badge/-Keras-black?style=for-the-badge&logo=keras&logoColor=white&color=D00000" />
    <img src="https://img.shields.io/badge/-FastAPI-black?style=for-the-badge&logo=fastapi&logoColor=white&color=009688" />
    <img src="https://img.shields.io/badge/-OpenCV-black?style=for-the-badge&logo=opencv&logoColor=white&color=5C3EE8" />
    <img src="https://img.shields.io/badge/-PostgreSQL-black?style=for-the-badge&logo=postgresql&logoColor=white&color=4169E1" />
    <img src="https://img.shields.io/badge/-NumPy-black?style=for-the-badge&logo=numpy&logoColor=white&color=013243" />
    <img src="https://img.shields.io/badge/-SQLAlchemy-black?style=for-the-badge&logo=sqlalchemy&logoColor=white&color=D71F00" />
  </div>

  <h1>😊 Projet Deep Learning – Détection d'Émotions Faciales</h1>
  <p><strong>Projet IA</strong> – Simplon Maghreb</p>
</div>

---

## 🧩 1. Introduction

L'objectif de ce projet est de concevoir un **système de détection d'émotions faciales en temps réel** utilisant un **réseau neuronal convolutif (CNN)** capable de classifier les expressions humaines en **7 catégories émotionnelles** : colère, dégoût, peur, joie, neutralité, tristesse et surprise.

Le projet comprend :
- **API REST** : Backend FastAPI pour l'analyse d'images
- **Modèle CNN** : Réseau de neurones pré-entraîné sur 48×48 pixels en niveaux de gris
- **Base de données** : PostgreSQL pour l'historique des prédictions
- **CI/CD** : Pipeline automatisé avec GitHub Actions
- **Détection faciale** : Haar Cascade OpenCV pour la localisation des visages

---

## ⚙️ 2. Modèle Utilisé

Le modèle est un **CNN (Convolutional Neural Network)** construit avec **TensorFlow/Keras** :

### Architecture
```
Input: (48, 48, 1) - Image en niveaux de gris

Conv2D Layers + Pooling (feature extraction)
Flatten Layer
Dense Layers (classification)
Output: Softmax (7 classes)
```

### Spécifications Techniques
- **Framework** : TensorFlow/Keras
- **Input Shape** : 48×48 pixels, grayscale
- **Normalisation** : Pixels /255 (valeurs entre 0 et 1)
- **Détection de visage** : Haar Cascade Classifier (OpenCV)
- **Nombre de classes** : 7 émotions

### Classes d'Émotions
| Index | Émotion     | Description                        |
|-------|-------------|------------------------------------|
| 0     | Angry       | Colère, frustration, irritation    |
| 1     | Disgusted   | Dégoût, répulsion                  |
| 2     | Fearful     | Peur, anxiété, inquiétude          |
| 3     | Happy       | Joie, bonheur, satisfaction        |
| 4     | Neutral     | Neutre, expression calme           |
| 5     | Sad         | Tristesse, chagrin                 |
| 6     | Surprised   | Surprise, étonnement, stupéfaction |

---

## 📈 3. Résultats Obtenus

### Performance du Système

| Métrique               | Valeur         |
|------------------------|----------------|
| **Test Coverage**      | **84%**        |
| **Détection de visage**| Haar Cascade   |
| **Format d'entrée**    | 48×48 grayscale|
| **Temps de réponse**   | < 1 seconde    |

### 🔍 Analyse des Résultats

- Le modèle CNN pré-entraîné offre des **prédictions en temps réel** avec une haute précision.
- La **détection faciale** via Haar Cascade permet une localisation rapide et fiable des visages.
- L'architecture est **optimisée pour l'inférence** avec des images de petite taille (48×48).
- Les **erreurs de classification** proviennent principalement des émotions proches (ex. : *Angry* vs *Disgusted*).
- Le système est **robuste** pour des images avec des visages clairement visibles.

### API Endpoints Disponibles

| Endpoint                    | Méthode | Description                              |
|----------------------------|---------|------------------------------------------|
| `/`                        | GET     | Health check de l'API                    |
| `/api/predict_emotion`     | POST    | Prédiction d'émotion depuis une image    |
| `/api/history`             | GET     | Historique des prédictions stockées      |

---

## ✅ 4. Justification du Modèle

Le **CNN** a été retenu pour sa **capacité à extraire des features spatiales** des images faciales, contrairement aux DNN classiques.

### Avantages Observés :
1. **Précision élevée** : Les convolutions capturent les patterns faciaux (sourcils, bouche, rides).
2. **Rapidité d'inférence** : Le modèle est optimisé pour des images 48×48, permettant un traitement rapide.
3. **Architecture éprouvée** : Les CNN dominent la classification d'images depuis AlexNet.
4. **Pipeline complet** : Le projet intègre détection faciale + classification + stockage.

### Architecture Backend
```
FastAPI (API REST)
    ↓
OpenCV (Détection de visage)
    ↓
TensorFlow/Keras (Prédiction d'émotion)
    ↓
PostgreSQL (Stockage de l'historique)
```

### Technologies Clés
- **FastAPI** : Framework web moderne et performant
- **SQLAlchemy** : ORM pour la gestion de la base de données
- **Uvicorn** : Serveur ASGI haute performance
- **OpenCV** : Bibliothèque de vision par ordinateur
- **TensorFlow** : Framework de Deep Learning

---

## 🧠 5. Pistes d'Amélioration

### Court Terme
- Ajouter **data augmentation** pour améliorer la robustesse (rotation, flip, zoom).
- Tester un **modèle plus profond** (ResNet, EfficientNet) pour améliorer la précision.
- Implémenter **l'authentification API** (OAuth2, JWT) pour sécuriser les endpoints.

### Moyen Terme
- Intégrer **Transfer Learning** avec des modèles pré-entraînés sur AffectNet ou FER-2013.
- Développer une **interface web interactive** (React, Vue.js) pour tester le modèle.
- Ajouter **l'analyse vidéo en temps réel** avec détection multi-visages.

### Long Terme
- Déployer sur **cloud** (AWS, GCP, Azure) avec conteneurisation Docker.
- Implémenter **des modèles multi-tâches** (émotion + âge + genre).
- Intégrer **des modèles Hugging Face** (Vision Transformer) pour benchmark.

---

## 🚀 6. Installation et Utilisation

### Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/OclaZ/brief_05_detection_demotions_faciales.git
cd brief_05_detection_demotions_faciales

# Créer un environnement virtuel
python -m venv myvenv312
myvenv312\Scripts\activate  # Windows
# source myvenv312/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Configurer la base de données (.env)
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=emotion_detection
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432

# Lancer l'API
uvicorn BACKEND.main:app --reload
```

### Utilisation de l'API

**Prédire une émotion :**
```bash
curl -X POST "http://localhost:8000/api/predict_emotion" \
  -F "file=@path/to/image.jpg"
```

**Récupérer l'historique :**
```bash
curl -X GET "http://localhost:8000/api/history"
```

**Documentation interactive :**
- Swagger UI : `http://localhost:8000/docs`
- ReDoc : `http://localhost:8000/redoc`

---

## 🧪 7. Tests et CI/CD

### Tests Automatisés

```bash
# Lancer tous les tests
pytest

# Avec rapport de couverture
pytest --cov=BACKEND --cov-report=term-missing
```

### Couverture de Code : **84%**

```
Name                            Stmts   Miss  Cover
-----------------------------------------------------
BACKEND/core/database.py          29      8    72%
BACKEND/main.py                   10      1    90%
BACKEND/models/predictions.py     11      1    91%
BACKEND/routes/predict.py         53      9    83%
BACKEND/schemas/schema.py         17      0   100%
-----------------------------------------------------
TOTAL                            120     19    84%
```

### CI/CD Pipeline (GitHub Actions)
- ✅ Tests automatiques sur chaque push/PR
- ✅ PostgreSQL service container pour les tests
- ✅ Rapport de couverture de code
- ✅ Support Python 3.12

---

## 🏁 8. Conclusion

Le prototype réalisé démontre la faisabilité d'un **système de détection d'émotions faciales en temps réel** avec un **CNN optimisé**.

### Points Clés :
✅ **Architecture complète** : API REST + ML + Database
✅ **Performance** : Inférence rapide et détection fiable
✅ **Qualité** : 84% de couverture de tests
✅ **Scalabilité** : Architecture prête pour le déploiement cloud
✅ **Extensibilité** : Facile d'ajouter de nouvelles fonctionnalités

Le projet constitue une **base solide** pour des applications de reconnaissance émotionnelle (chatbots empathiques, analyse de satisfaction client, systèmes de sécurité, etc.).

---

## 📚 Documentation Complète

Pour plus de détails techniques, consultez [DOCUMENTATION.md](DOCUMENTATION.md) :
- Installation détaillée
- Architecture du modèle
- Schéma de base de données
- Guide des endpoints API
- Guide de contribution

---

<div align="center">
  <p>👨‍💻 Projet réalisé par <strong><a href="https://github.com/OclaZ">OclaZ</a></strong> | Simplon Maghreb</p>
  <p>🔗 <a href="https://github.com/OclaZ/brief_05_detection_demotions_faciales">GitHub Repository</a></p>
</div>
