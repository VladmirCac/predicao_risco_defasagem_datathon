"""
Treina o RandomForestClassifier e salva artefatos em app/model:
- random_forest_abandono.joblib
- random_forest_abandono.pkl
- features.pkl
- threshold.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

TARGET = "ABANDONO"
YEAR_COL = "ANO"
DEFAULT_RECALL = 0.60


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_parquet(path)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(df[YEAR_COL].unique()) != {2022, 2023}:
        raise ValueError("A coluna ANO deve conter apenas 2022 e 2023 para treino/teste.")
    train = df[df[YEAR_COL] == 2022].copy()
    test = df[df[YEAR_COL] == 2023].copy()
    return train, test


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        class_weight={0: 1, 1: 2},
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def select_threshold_by_recall(
    recall: np.ndarray, thresholds: np.ndarray, min_recall: float
) -> float:
    idx = np.where(recall >= min_recall)[0]
    if len(idx) == 0:
        return 0.5
    last = idx[-1]
    return thresholds[last - 1] if last > 0 else thresholds[0]


def save_artifacts(model, features, threshold: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / "random_forest_abandono.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)

    joblib_path = out_dir / "random_forest_abandono.joblib"
    joblib.dump(model, joblib_path)

    features_path = out_dir / "features.pkl"
    with features_path.open("wb") as f:
        pickle.dump(features, f)

    threshold_path = out_dir / "threshold.pkl"
    with threshold_path.open("wb") as f:
        pickle.dump(threshold, f)


def main(data_path: Path, min_recall: float, model_dir: Path) -> None:
    df = load_dataset(data_path)
    print(f"Shape do dataset: {df.shape}")
    print("\nTipos de dados:\n")
    print(df.dtypes)

    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada.")

    features = [c for c in df.columns if c not in [TARGET, YEAR_COL]]
    train, test = split_train_test(df)

    X_train, y_train = sanitize_features(train[features]), train[TARGET]
    X_test, y_test = sanitize_features(test[features]), test[TARGET]

    model = train_model(X_train, y_train)
    print(f"OOB score: {model.oob_score_:.4f}")

    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    prec, rec, thr = precision_recall_curve(y_test, proba)
    threshold = select_threshold_by_recall(rec, thr, min_recall)
    preds = (proba >= threshold).astype(int)

    print(f"\nThreshold usado: {threshold:.4f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=3))
    print(f"\nAcionados: {preds.sum()}/{len(preds)} = {preds.mean():.1%}")

    save_artifacts(model, features, threshold, model_dir)
    print(f"\nArtefatos salvos em {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelo de abandono e salva artefatos.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root() / "data" / "processed" / "df_trein.parquet",
        help="Caminho do arquivo parquet processado.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=DEFAULT_RECALL,
        help="Recall mínimo para definir o threshold de disparo.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root() / "app" / "model",
        help="Diretório de saída dos artefatos do modelo.",
    )
    args = parser.parse_args()
    main(args.data_path, args.min_recall, args.model_dir)
