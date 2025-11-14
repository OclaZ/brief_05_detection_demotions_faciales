from fastapi.testclient import TestClient
from BACKEND.main import app

client = TestClient(app)

def test_prediction():
    # Open the file inside the POST call (prevents "closed file" errors)
    with open("tests/test_face.png", "rb") as f:
        files = {"file": ("test_face.png", f, "image/png")}
        response = client.post("/api/predict_emotion", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "emotion" in data
    assert "confidence" in data
    assert isinstance(data["emotion"], str)
    assert isinstance(data["confidence"], float)
