"""
Feature engineering pipeline.

Lê o parquet refinado produzido em `src/preprocessing.py`, calcula o alvo
ABANDONO, imputa nulos e salva a base final de treino em
`data/processed/df_trein.parquet`.

O script não imprime análises; apenas executa a transformação dos dados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFINED_PARQUET = ROOT / "data" / "refined" / "df_base.parquet"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "df_trein.parquet"

# Anos com rastro de presença. 2024 é o último ano disponível e não permite
# observar abandono no ano seguinte.
ANO_MAX_DEFAULT = 2024

# Ordem de colunas esperada pelo restante do fluxo.
TRAIN_COL_ORDER: Sequence[str] = (
    "ANO",
    "IDADE",
    "FASE",
    "DEFASAGEM",
    "IAA",
    "IEG",
    "IDA",
    "IAN",
    "IPS",
    "IPV",
    "NOTA_MAT",
    "NOTA_POR",
    "ABANDONO",
)


def build_abandono_flags(df: pd.DataFrame, ano_max: int = ANO_MAX_DEFAULT) -> pd.DataFrame:
    """Adiciona colunas de presença e alvo ABANDONO, replicando o notebook."""
    data = df.copy()

    # Considera apenas fases válidas no histórico.
    data = data[data["FASE"].between(1, 8)].copy()
    data = data.sort_values(["RA", "ANO"])

    data["PRESENTE_ANO_SEGUINTE"] = (
        data.groupby("RA")["ANO"].shift(-1).eq(data["ANO"] + 1)
    ).fillna(False)

    data["OBSERVAVEL_ABANDONO"] = (data["ANO"] < ano_max).astype(int)

    data["ABANDONO"] = (
        (data["FASE"] < 8)
        & (data["OBSERVAVEL_ABANDONO"] == 1)
        & (~data["PRESENTE_ANO_SEGUINTE"])
    ).astype(int)

    return data


def _validate_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}")


def imputar_nulos_por_fase(
    df_train: pd.DataFrame,
    fase_col: str = "FASE",
    cols_imputar: Sequence[str] = ("IDA", "IEG", "IPV", "IPS", "NOTA_MAT", "NOTA_POR"),
    add_missing_flags: bool = True,
    suffix_missing: str = "_MISSING",
) -> pd.DataFrame:
    """
    Imputa nulos numéricos pela mediana de cada FASE com fallback global.
    Opcionalmente adiciona flags de missing (0/1) por coluna imputada.
    """
    df = df_train.copy()

    if fase_col not in df.columns:
        raise ValueError(f"Coluna '{fase_col}' não existe no dataframe.")

    cols_existentes = [c for c in cols_imputar if c in df.columns]

    if add_missing_flags:
        for col in cols_existentes:
            df[f"{col}{suffix_missing}"] = df[col].isna().astype(int)

    for col in cols_existentes:
        global_med = df[col].median()
        med_por_fase = df.groupby(fase_col)[col].median()

        def _fill_group(series: pd.Series) -> pd.Series:
            med = med_por_fase.get(series.name)
            if pd.isna(med):
                med = global_med
            return series.fillna(med)

        df[col] = df.groupby(fase_col, group_keys=False)[col].apply(_fill_group)

        # Fallback final: se ainda restar NaN (caso coluna inteira seja NaN), zera.
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


def build_training_set(df: pd.DataFrame, ano_max: int = ANO_MAX_DEFAULT) -> pd.DataFrame:
    """
    Cria base final de treino:
      - anos observáveis (ANO <= ano_max-1)
      - fases 1 a 7
      - remoção de colunas de identificação/vazamento
      - imposição de dtypes e ordem de colunas
      - imputação de nulos por fase
    """
    df_abandono = build_abandono_flags(df, ano_max=ano_max)

    df_train = df_abandono[
        (df_abandono["OBSERVAVEL_ABANDONO"] == 1) & (df_abandono["FASE"].between(1, 7))
    ].copy()

    cols_to_drop = ["RA", "PRESENTE_ANO_SEGUINTE", "OBSERVAVEL_ABANDONO"]
    df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])

    # Tipagens base.
    df_train["ANO"] = df_train["ANO"].astype("int64")
    df_train["FASE"] = df_train["FASE"].astype("Int64")
    df_train["IDADE"] = df_train["IDADE"].round().astype("Int64")
    df_train["DEFASAGEM"] = df_train["DEFASAGEM"].round().astype("Int64")
    df_train["ABANDONO"] = df_train["ABANDONO"].astype("int64")

    _validate_columns(df_train, TRAIN_COL_ORDER)
    df_train = df_train[list(TRAIN_COL_ORDER)]

    df_train_imp = imputar_nulos_por_fase(df_train)

    # Após imputação, garante inteiros densos para colunas conceituais.
    int_cols = ["IDADE", "FASE", "DEFASAGEM"]
    for col in int_cols:
        if df_train_imp[col].isna().any():
            # Se algo ficou NaN, converte para 0 para manter compatibilidade com astype.
            df_train_imp[col] = df_train_imp[col].fillna(0)
        df_train_imp[col] = df_train_imp[col].astype("int64")

    return df_train_imp


def run_pipeline(
    refined_path: Path = REFINED_PARQUET,
    output_path: Path = PROCESSED_PARQUET,
    ano_max: int = ANO_MAX_DEFAULT,
) -> pd.DataFrame:
    """Executa a engenharia de features end-to-end e salva o parquet final."""
    if not refined_path.exists():
        raise FileNotFoundError(f"Arquivo refinado não encontrado: {refined_path}")

    df_refined = pd.read_parquet(refined_path)
    df_processed = build_training_set(df_refined, ano_max=ano_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    return df_processed


if __name__ == "__main__":
    run_pipeline()
