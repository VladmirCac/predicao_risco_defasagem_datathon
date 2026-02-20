"""
Gera um dashboard HTML de data drift usando Evidently.

Uso:
    python -m monitoring.drift_report \\
        --reference data/processed/df_trein.parquet \\
        --current data/refined/df_base.parquet \\
        --output monitoring/drift_report.html

O arquivo HTML gerado pode ser servido pela rota GET /monitoring da API.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import numpy as np

# Preferimos a API legacy (0.7.x) porque suporta column_mapping em Report.run().
try:  # Evidently 0.7.x legacy
    from evidently.legacy.pipeline.column_mapping import ColumnMapping  # type: ignore
    from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
    from evidently.legacy.report import Report  # type: ignore
except Exception:
    # Fallback para APIs mais novas/antigas.
    try:
        from evidently import ColumnMapping, Report  # type: ignore
    except Exception:
        from evidently.utils.data_operations import ColumnMapping  # type: ignore
        from evidently.legacy.report import Report  # type: ignore
    try:
        from evidently.presets import DataDriftPreset  # type: ignore
    except Exception:
        from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("monitoring")


def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    logger.info("Lendo dataset: %s", path)
    return pd.read_parquet(path)


def _build_column_mapping(
    target: Optional[str],
    prediction: Optional[str],
    id_columns: Iterable[str] | None,
    columns: Iterable[str],
):
    """
    Constrói ColumnMapping inferindo numéricas como o restante das colunas.
    Necessário para Evidently 0.7.x (sem column_mapping em run).
    """
    id_cols = list(id_columns or [])
    cols = list(columns)
    numerical = [c for c in cols if c not in id_cols and c != target and c != prediction]
    categorical: list[str] = []
    return ColumnMapping(
        target=target,
        prediction=prediction,
        id=id_cols[0] if id_cols else None,
        numerical_features=numerical,
        categorical_features=categorical,
    )


def generate_drift_report(reference: Path, current: Path, output: Path, target: str | None, prediction: str | None, id_columns: list[str]):
    ref_df = _load_df(reference)
    cur_df = _load_df(current)

    # Usa apenas colunas presentes em ambas as bases para evitar colunas vazias.
    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[common_cols]
    cur_df = cur_df[common_cols].replace({pd.NA: np.nan})

    # Target/prediction só são usados se estiverem em ambas.
    if target not in common_cols:
        target = None
    if prediction not in common_cols:
        prediction = None

    mapping = _build_column_mapping(target, prediction, id_columns, ref_df.columns)

    preset = DataDriftPreset()
    report = Report(metrics=[preset])
    logger.info("Executando cálculo de drift...")
    try:
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=mapping)
    except TypeError:
        # Algumas versões do Evidently não aceitam column_mapping em run; já passamos no preset.
        report.run(reference_data=ref_df, current_data=cur_df)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))
    logger.info("Dashboard salvo em: %s", output.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Gera relatório de data drift com Evidently.")
    parser.add_argument("--reference", type=Path, default=Path("data/processed/df_trein.parquet"), help="Dataset de referência (parquet).")
    parser.add_argument("--current", type=Path, default=Path("data/refined/df_base.parquet"), help="Dataset atual (parquet).")
    parser.add_argument("--output", type=Path, default=Path("monitoring/drift_report.html"), help="Caminho do dashboard HTML.")
    parser.add_argument("--target", type=str, default=None, help="Nome da coluna de target, se houver.")
    parser.add_argument("--prediction", type=str, default=None, help="Nome da coluna de predição, se houver.")
    parser.add_argument(
        "--id-column",
        dest="id_columns",
        action="append",
        default=[],
        help="Coluna(s) de identificação para ignorar no drift. Pode repetir a flag.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_drift_report(
        reference=args.reference,
        current=args.current,
        output=args.output,
        target=args.target,
        prediction=args.prediction,
        id_columns=args.id_columns,
    )


if __name__ == "__main__":
    main()
