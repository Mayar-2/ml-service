import logging, os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
state = {"model": None, "requests": 0}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.warning("Model not found at %s", MODEL_PATH)
    yield

app = FastAPI(title="ML Service", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    state["requests"] += 1
    try:
        pred = state["model"].predict([req.features])[0]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return PredictResponse(prediction=int(pred))

@app.get("/metrics")
def metrics():
    return {"requests_total": state["requests"]}