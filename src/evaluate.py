"""
Avalia o modelo RandomForest salvo em `app/model` usando o hold-out temporal
(ANO==2023) presente em `data/processed/df_trein.parquet`.

Métricas impressas:
- ROC-AUC e PR-AUC
- Matriz de confusão e classification_report com o threshold salvo

Opcionalmente salva um JSON com as métricas.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

TARGET = "ABANDONO"
YEAR_COL = "ANO"
DEFAULT_TEST_YEAR = 2023


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa inf/nan de maneira estável para predição."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(path)


def load_artifacts(model_dir: Path):
    model_path = model_dir / "random_forest_abandono.joblib"
    features_path = model_dir / "features.pkl"
    threshold_path = model_dir / "threshold.pkl"

    model = joblib.load(model_path)
    with features_path.open("rb") as f:
        features = pickle.load(f)
    with threshold_path.open("rb") as f:
        threshold = pickle.load(f)

    return model, features, float(threshold)


def evaluate_model(
    df: pd.DataFrame,
    model,
    features: list[str],
    threshold: float,
    test_year: int = DEFAULT_TEST_YEAR,
):
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada.")
    if YEAR_COL not in df.columns:
        raise ValueError(f"Coluna '{YEAR_COL}' não encontrada.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no dataset para avaliação: {missing}")

    test = df[df[YEAR_COL] == test_year].copy()
    if test.empty:
        raise ValueError(f"Nenhuma linha encontrada para ANO == {test_year}.")

    X_test = sanitize_features(test[features])
    y_test = test[TARGET]

    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    prec, rec, thr = precision_recall_curve(y_test, proba)

    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=3)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "test_year": int(test_year),
        "support": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "precision_curve": prec.tolist(),
        "recall_curve": rec.tolist(),
        "thresholds_curve": thr.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    return metrics, report


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def main(data_path: Path, model_dir: Path, test_year: int, metrics_out: Path | None):
    df = load_dataset(data_path)
    model, features, threshold = load_artifacts(model_dir)

    metrics, report_txt = evaluate_model(df, model, features, threshold, test_year)

    print(f"Dataset usado: {data_path} (linhas={len(df)})")
    print(f"Modelo carregado de: {model_dir}")
    print(f"Threshold aplicado: {metrics['threshold']:.4f}\n")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC : {metrics['pr_auc']:.4f}\n")
    print("Matriz de confusão:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:")
    print(report_txt)

    if metrics_out:
        save_metrics(metrics, metrics_out)
        print(f"\nMétricas salvas em {metrics_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o modelo salvo.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root() / "data" / "processed" / "df_trein.parquet",
        help="Caminho do parquet processado contendo coluna ANO e target ABANDONO.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root() / "app" / "model",
        help="Diretório com random_forest_abandono.joblib, features.pkl e threshold.pkl.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=DEFAULT_TEST_YEAR,
        help="Ano a ser usado como hold-out para avaliação (padrão: 2023).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Se informado, salva um JSON com métricas neste caminho.",
    )
    args = parser.parse_args()
    main(args.data_path, args.model_dir, args.test_year, args.metrics_out)
