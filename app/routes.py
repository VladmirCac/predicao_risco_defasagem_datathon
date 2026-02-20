import logging
import pickle
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger("app.api")

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "random_forest_abandono.joblib"
FEATURES_PATH = MODEL_DIR / "features.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"


class PredictionRequest(BaseModel):
    """
    Recebe as features usadas pelo modelo.
    Campos podem vir ausentes; flags *_MISSING são derivadas automaticamente.
    """

    payload: Dict[str, float | None] = Field(
        ..., description="Mapa feature -> valor para cada feature esperada (ausentes são tratados)."
    )


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    threshold: float


def _load_artifacts():
    if not (MODEL_PATH.exists() and FEATURES_PATH.exists() and THRESHOLD_PATH.exists()):
        raise FileNotFoundError("Artefatos do modelo não encontrados em app/model.")
    model = joblib.load(MODEL_PATH)
    with FEATURES_PATH.open("rb") as f:
        features: List[str] = pickle.load(f)
    with THRESHOLD_PATH.open("rb") as f:
        threshold: float = pickle.load(f)
    return model, features, float(threshold)


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def _build_feature_row(payload: Dict[str, float | None], features: List[str]) -> Dict[str, float]:
    """
    Garante presença de todas as features esperadas.
    - Se feature base estiver faltando, usa NaN e liga flag *_MISSING.
    - Flags *_MISSING são calculadas se não vierem no payload.
    """
    row: Dict[str, float] = {}
    for feat in features:
        if feat.endswith("_MISSING"):
            base = feat[: -len("_MISSING")]
            # Usa flag enviada, se houver; senão deriva do valor base (1 se ausente ou NaN).
            if feat in payload:
                row[feat] = float(payload[feat]) if payload[feat] is not None else 1.0
            else:
                base_val = payload.get(base)
                row[feat] = 1.0 if base_val is None or pd.isna(base_val) else 0.0
        else:
            row[feat] = payload.get(feat)
    return row


try:
    MODEL, FEATURES, THRESHOLD = _load_artifacts()
except FileNotFoundError as exc:
    MODEL = FEATURES = THRESHOLD = None
    LOAD_ERROR = exc
    logger.exception("Erro ao carregar artefatos do modelo")
else:
    LOAD_ERROR = None
    logger.info(
        "Artefatos do modelo carregados",
        extra={"n_features": len(FEATURES) if FEATURES else 0, "threshold": THRESHOLD},
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if LOAD_ERROR:
        logger.error("Tentativa de predição sem artefatos carregados")
        raise HTTPException(status_code=500, detail=str(LOAD_ERROR))

    # Monta linha completa, derivando flags de missing quando necessário
    row = _build_feature_row(request.payload, FEATURES)

    X = pd.DataFrame([row])
    X = _sanitize_frame(X)

    proba = float(MODEL.predict_proba(X)[:, 1][0])
    pred = int(proba >= THRESHOLD)

    logger.info(
        "Predição executada",
        extra={
            "probability": round(proba, 4),
            "prediction": pred,
            "threshold": THRESHOLD,
            "features_missing": sum(k.endswith("_MISSING") and v == 1.0 for k, v in row.items()),
        },
    )

    return PredictionResponse(probability=proba, prediction=pred, threshold=THRESHOLD)
