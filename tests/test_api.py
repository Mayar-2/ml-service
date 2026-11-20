import pytest
from fastapi.testclient import TestClient
from app.main import app, state

class FakeModel:
    def predict(self, X):
        return [1]

class BrokenModel:
    def predict(self, X):
        raise RuntimeError("boom")

@pytest.fixture
def client():
    state["model"], state["requests"] = FakeModel(), 0
    return TestClient(app)  # without "with" -> lifespan not triggered

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True

def test_predict_valid(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200 and r.json() == {"prediction": 1}

def test_predict_too_few_features(client):
    assert client.post("/predict", json={"features": [1, 2]}).status_code == 422

def test_predict_model_not_loaded(client):
    state["model"] = None
    assert client.post("/predict", json={"features": [1, 2, 3, 4]}).status_code == 503

def test_predict_model_error(client):
    state["model"] = BrokenModel()
    assert client.post("/predict", json={"features": [1, 2, 3, 4]}).status_code == 500

def test_health_model_not_loaded(client):
    state["model"] = None
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False

def test_predict_too_many_features(client):
    r = client.post("/predict", json={"features": [1, 2, 3, 4, 5]})
    assert r.status_code == 422

def test_predict_non_numeric_values(client):
    r = client.post("/predict", json={"features": ["abc", "b", "c", "d"]})
    assert r.status_code == 422

def test_predict_empty_body(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422

def test_metrics_returns_200(client):
    r = client.get("/metrics")
    assert r.status_code == 200

def test_metrics_counter_increments(client):
    client.post("/predict", json={"features": [1, 2, 3, 4]})
    client.post("/predict", json={"features": [1, 2, 3, 4]})
    r = client.get("/metrics")
    assert r.json()["requests_total"] == 2

def test_unknown_path_returns_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404

def test_prediction_is_int(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert isinstance(r.json()["prediction"], int)