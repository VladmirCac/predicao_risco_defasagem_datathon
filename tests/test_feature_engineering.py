import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    build_abandono_flags,
    imputar_nulos_por_fase,
    build_training_set,
)


def make_base_df():
    return pd.DataFrame(
        {
            "RA": ["1", "1", "2", "2"],
            "ANO": [2022, 2023, 2022, 2023],
            "FASE": [1, 2, 1, 1],
            "IDADE": [10, 11, 9, np.nan],
            "DEFASAGEM": [0, 1, 0, 0],
            "IAA": [5, 6, 7, 8],
            "IEG": [1, np.nan, 2, 2],
            "IDA": [3, 4, np.nan, 1],
            "IAN": [1, 2, 2, 3],
            "IPS": [2, 2, 2, np.nan],
            "IPV": [3, 3, 3, 3],
            "NOTA_MAT": [7, 8, np.nan, 6],
            "NOTA_POR": [6, 7, 6, 5],
        }
    )


def test_build_abandono_flags_marks_missing_next_year():
    df = pd.DataFrame(
        {
            "RA": ["1", "1", "2"],
            "ANO": [2022, 2023, 2022],
            "FASE": [1, 2, 7],
        }
    )
    out = build_abandono_flags(df, ano_max=2024)
    abandono = out[out["RA"] == "2"]["ABANDONO"].iloc[0]
    assert abandono == 1  # fase<8 e sem presença no ano seguinte


def test_imputar_nulos_por_fase_uses_median_and_flags():
    df = pd.DataFrame(
        {
            "FASE": [1, 1, 2],
            "IEG": [np.nan, 2, np.nan],
            "NOTA_MAT": [1, np.nan, np.nan],
        }
    )
    out = imputar_nulos_por_fase(df)
    # FASE 1 median for IEG =2, fallback para FASE 2 usa mediana global (=2)
    assert out.loc[2, "IEG"] == 2
    # Flags de missing criadas
    assert out["IEG_MISSING"].tolist() == [1, 0, 1]
    assert out["NOTA_MAT_MISSING"].tolist() == [0, 1, 1]


def test_build_training_set_filters_and_types():
    base_df = make_base_df()
    train_df = build_training_set(base_df, ano_max=2024)

    assert set(train_df["ANO"].unique()) == {2022, 2023}
    assert train_df["ABANDONO"].dtype == "int64"
    assert all(train_df["FASE"].between(1, 7))
    # ordem das colunas garantida
    expected_first_cols = ["ANO", "IDADE", "FASE", "DEFASAGEM"]
    assert list(train_df.columns[:4]) == expected_first_cols
