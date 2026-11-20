# ML Service (AI381 – Production-Grade ML Service)

A small, reliable ML prediction service built with **FastAPI**, covered by automated tests and packaged with **Docker**.
This README is built up by the whole team; each member documents their own part.

---

## Part A – The Application, Tests and Docker image

### What this part does
- Serves a trained scikit-learn model through a REST API (FastAPI).
- Validates every request (Pydantic) and fails with clear error codes.
- Is covered by 13 automated tests (pytest).
- Is packaged as a small, secure Docker image (multi-stage, non-root user, healthcheck).

### Project structure (Part A)
```
app/
  __init__.py
  main.py          # FastAPI app: /health, /predict, /metrics
  schemas.py       # Pydantic request/response models
tests/
  test_api.py      # 13 pytest tests
scripts/
  make_dummy_model.py   # creates a placeholder model (iris + LogisticRegression)
models/
  model.joblib     # model file loaded at startup
Dockerfile
.dockerignore
pytest.ini
requirements.txt
```

### Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# only if models/model.joblib does not exist yet
python scripts/make_dummy_model.py

uvicorn app.main:app --reload --port 8001
```
Open http://localhost:8001/docs for the interactive API documentation.

> Port 8001 is used locally only to avoid clashing with other services on port 8000. Inside Docker the app always listens on 8000.

### Configuration
| Variable     | Default                | Meaning                          |
|--------------|------------------------|----------------------------------|
| `MODEL_PATH` | `models/model.joblib`  | Path of the model file to load   |

### API

| Method | Path       | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Liveness check, also tells if the model is loaded |
| POST   | `/predict` | Returns a prediction for 4 input features     |
| GET    | `/metrics` | Number of predictions served so far           |

**Health**
```bash
curl http://localhost:8001/health
# {"status":"ok","model_loaded":true}
```

**Predict**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
# {"prediction":0}
```

**Error codes**
| Code | When                                                        |
|------|-------------------------------------------------------------|
| 422  | Invalid input (wrong number of features, non-numeric, empty body) |
| 503  | The model is not loaded                                     |
| 500  | The model failed while predicting                           |
| 404  | Unknown path                                                |

### How the app works
1. On startup (`lifespan`) the app loads the model from `MODEL_PATH` into memory once, so requests are fast. If the file is missing, the app still starts and logs a warning; `/health` then reports `model_loaded: false` and `/predict` returns 503.
2. `/predict` validates the body with Pydantic (`features` must be exactly 4 numbers), calls `model.predict`, and returns the class as an `int`.
3. Every successful prediction increments a counter exposed on `/metrics`.
4. Errors are logged with `logger.exception` so problems can be debugged from the logs.

### Tests
```bash
pytest -v
```
13 tests, all running without a real model file (a `FakeModel` is injected), so they are fast and deterministic:

- `/health`: model loaded and model not loaded
- `/predict`: valid input, too few features, too many features, non-numeric values, empty body
- Failure handling: model not loaded (503), model raises an error (500)
- `/metrics`: returns 200 and the counter increments per prediction
- Unknown path returns 404
- The prediction type is `int`

`pytest.ini` sets `pythonpath = .` so tests can import the `app` package, locally and in CI.

### Docker
```bash
docker build -t ml-service .
docker run -p 8001:8000 ml-service
curl http://localhost:8001/health
```

Design choices in the `Dockerfile`:
- **Multi-stage build**: dependencies are installed in a builder stage, and only the installed packages are copied to the final image, which keeps it small.
- **Slim base image** (`python:3.11-slim`).
- **Non-root user** (`appuser`) for security.
- **`HEALTHCHECK`** calling `/health`, so Docker (and later Kubernetes) knows if the service is healthy.
- **`.dockerignore`** keeps `.venv`, tests, caches and `.git` out of the image, making builds faster and the image smaller.

### Design decisions and trade-offs
- **FastAPI + Pydantic**: automatic validation and free interactive docs at `/docs`.
- **Model loaded once at startup** instead of per request: faster, but the model is fixed until the service restarts.
- **Simple in-memory counter for `/metrics`**: easy and enough for this project; it resets on restart and is not shared across replicas (a real system would use Prometheus).
- **Placeholder model**: Part A uses a dummy iris model so the service can be developed and tested independently; the real model comes from the MLflow part.

---

## Part B – ML tracking, promotion gate and CI
_To be written by Person B._

## Part C – Kubernetes (minikube), GitOps, runbook and cost
_To be written by Person C._
