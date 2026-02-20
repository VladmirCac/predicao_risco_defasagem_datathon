import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

from src.train import (
    split_train_test,
    sanitize_features,
    select_threshold_by_recall,
    train_model,
)
from src.evaluate import evaluate_model


def small_dataset():
    data = []
    rng = np.random.default_rng(42)
    for year in [2022, 2023]:
        for i in range(30):
            data.append(
                {
                    "ANO": year,
                    "f1": rng.normal(),
                    "f2": rng.normal(),
                    "ABANDONO": rng.integers(0, 2),
                }
            )
    return pd.DataFrame(data)


def test_split_train_test_enforces_years():
    df = pd.DataFrame({"ANO": [2022, 2023], "ABANDONO": [0, 1]})
    train, test = split_train_test(df)
    assert train["ANO"].nunique() == 1 and test["ANO"].nunique() == 1

    bad_df = pd.DataFrame({"ANO": [2022, 2024], "ABANDONO": [0, 1]})
    with pytest.raises(ValueError):
        split_train_test(bad_df)


def test_select_threshold_by_recall_returns_last_valid():
    prec = np.array([1, 0.8, 0.6])
    rec = np.array([0.3, 0.65, 0.7])
    thr = np.array([0.9, 0.8])
    # PR curve from sklearn has len(thresholds)=len(prec)-1
    selected = select_threshold_by_recall(rec, thr, min_recall=0.6)
    assert selected == thr[-1]


def test_train_and_evaluate_end_to_end(tmp_path):
    df = small_dataset()
    train_df, test_df = split_train_test(df)

    X_train = sanitize_features(train_df[["f1", "f2"]])
    y_train = train_df["ABANDONO"]
    model = train_model(X_train, y_train)

    # simulate precision-recall curve to pick threshold >= default recall
    proba = model.predict_proba(sanitize_features(test_df[["f1", "f2"]]))[:, 1]
    prec, rec, thr = precision_recall_curve(test_df["ABANDONO"], proba)
    threshold = select_threshold_by_recall(rec, thr, 0.5)

    metrics, _ = evaluate_model(
        df,
        model,
        features=["f1", "f2"],
        threshold=threshold,
        test_year=2023,
    )

    assert set(metrics.keys()) >= {"roc_auc", "pr_auc", "threshold", "confusion_matrix"}
    assert metrics["test_year"] == 2023
