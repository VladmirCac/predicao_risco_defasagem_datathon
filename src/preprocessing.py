"""
Pipeline de pré-processamento para consolidar a base PEDE (2022–2024) a partir
do Excel bruto em `data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`.

Saída principal: `data/refined/df_base.parquet`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_EXCEL = ROOT / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
REFINED_DIR = ROOT / "data" / "refined"
REFINED_PARQUET = REFINED_DIR / "df_base.parquet"


def normalize_col(col: str) -> str:
    """Normaliza nomes de colunas para snake_case ASCII sem símbolos."""
    col = str(col).strip()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    col = col.lower()
    col = re.sub(r"[\\/\\.\\-]+", "_", col)
    col = re.sub(r"\\s+", "_", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def coalesce_cols(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Retorna a primeira coluna existente em `candidates`, preenchendo para NaN se nenhuma aparecer."""
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[cols].bfill(axis=1).iloc[:, 0]


def parse_idade(valor) -> float:
    """
    Limpa idade e devolve float ou NaN.
    - Números plausíveis (0<idade<120) são mantidos.
    - Datas 1900/1899-mm-dd (erro comum do Excel) viram o dia (8 -> 8 anos).
    - Datas muito antigas (<1920) também usam o dia como fallback.
    """
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor) if 0 < float(valor) < 120 else np.nan

    dt = pd.to_datetime(valor, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        if dt.year in {1899, 1900} or dt.year < 1920:
            return float(dt.day)

    try:
        v = float(str(valor).replace(",", "."))
        return v if 0 < v < 120 else np.nan
    except Exception:
        return np.nan


def parse_fase_int(valor):
    """
    Converte FASE para inteiro seguindo regras:
    1) Números puros -> inteiro.
    2) Número + letra (1A, 2b) -> número.
    3) Contendo 'alfa'/'alpha' -> 0.
    4) 'fase X' -> X.
    Retorna pd.NA se não conseguir.
    """
    try:
        if valor != valor:
            return pd.NA
    except Exception:
        pass

    if isinstance(valor, int):
        return valor
    if isinstance(valor, float):
        return int(valor) if valor.is_integer() else pd.NA

    s = str(valor).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA

    sl = s.lower()
    if sl.isdigit():
        return int(sl)
    if sl.replace(".", "", 1).isdigit() and sl.endswith(".0"):
        try:
            return int(float(sl))
        except Exception:
            return pd.NA
    if len(sl) >= 2 and sl[:-1].isdigit() and sl[-1].isalpha():
        return int(sl[:-1])
    if "alfa" in sl or "alpha" in sl:
        return 0
    if "fase" in sl:
        num = ""
        for c in sl:
            if c.isdigit():
                num += c
            elif num:
                break
        if num:
            return int(num)
    return pd.NA


def build_features_base_raw(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona e limpa colunas essenciais para modelagem.
    Não cria variáveis derivadas; apenas organiza e clippa ruído.
    """
    df = df_all.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    if "ra" not in df.columns:
        raise ValueError("Coluna RA não encontrada após normalização.")
    if "ano" not in df.columns:
        raise ValueError("Coluna ANO não encontrada após normalização.")

    candidates = {
        "idade": [
            "idade",
            "idade_22",
            "idade22",
            "idade_22_anos",
            "idade_2022",
            "idade_2023",
            "idade_2024",
            "idade_aluno",
        ],
        "fase": ["fase", "fase_2022", "fase_2023", "fase_2024"],
        "defasagem": ["defasagem", "defasagem_2021", "defasagem_2022", "defasagem_2023", "defasagem_2024", "defas"],
        "iaa": ["iaa", "iaa_2022", "iaa_2023", "iaa_2024"],
        "ieg": ["ieg", "ieg_2022", "ieg_2023", "ieg_2024"],
        "ida": ["ida", "ida_2022", "ida_2023", "ida_2024"],
        "ian": ["ian", "ian_2022", "ian_2023", "ian_2024"],
        "ips": ["ips", "ips_2022", "ips_2023", "ips_2024"],
        "ipv": ["ipv", "ipv_2022", "ipv_2023", "ipv_2024"],
        "nota_mat": ["nota_mat", "nota_mat_2022", "nota_mat_2023", "nota_mat_2024", "mat", "matem"],
        "nota_por": ["nota_por", "nota_port", "nota_por_2022", "nota_por_2023", "nota_por_2024", "por", "portug"],
    }

    out = pd.DataFrame(index=df.index)
    out["RA"] = df["ra"].astype(str).str.strip()
    out["ANO"] = pd.to_numeric(df["ano"], errors="coerce")

    out["IDADE"] = coalesce_cols(df, candidates["idade"]).apply(parse_idade)
    out["FASE"] = coalesce_cols(df, candidates["fase"]).apply(parse_fase_int).astype("Int64")

    out["DEFASAGEM"] = pd.to_numeric(coalesce_cols(df, candidates["defasagem"]), errors="coerce")

    for k in ["iaa", "ieg", "ida", "ian", "ips", "ipv"]:
        out[k.upper()] = pd.to_numeric(coalesce_cols(df, candidates[k]), errors="coerce")

    out["NOTA_MAT"] = pd.to_numeric(coalesce_cols(df, candidates["nota_mat"]), errors="coerce")
    out["NOTA_POR"] = pd.to_numeric(coalesce_cols(df, candidates["nota_por"]), errors="coerce")

    out = out.dropna(subset=["RA", "ANO"])

    for col in ["IAA", "IEG", "IDA", "IAN", "IPS", "IPV", "NOTA_MAT", "NOTA_POR", "NOTA_ING"]:
        if col in out.columns:
            out[col] = out[col].clip(lower=0, upper=10)

    features_base_raw = [
        "RA",
        "ANO",
        "IDADE",
        "FASE",
        "IAA",
        "IEG",
        "IDA",
        "IAN",
        "IPS",
        "IPV",
        "NOTA_MAT",
        "NOTA_POR",
        "DEFASAGEM",
    ]
    return out[features_base_raw]


def read_raw_excel(path: Path = RAW_EXCEL) -> dict[str, pd.DataFrame]:
    """Lê todas as abas do Excel bruto e devolve dicionário {aba: DataFrame}."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_excel(path, sheet_name=None)


def concat_sheets_with_year(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Adiciona coluna ANO extraída do nome da aba e concatena todas."""
    dfs = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        year = extract_year_from_sheet_name(sheet_name)
        if year is None:
            print(f"[WARN] Aba ignorada (sem ano identificável): {sheet_name}")
            continue
        df["ANO"] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def extract_year_from_sheet_name(sheet_name) -> int | None:
    """
    Extrai ano de um nome de aba.
    - Procura 4 dígitos primeiro (ex: 2022)
    - Se achar apenas 2 dígitos (ex: 22), assume 2000+valor
    Retorna None se não encontrar.
    """
    s = str(sheet_name)
    matches = re.findall(r"\d+", s)
    if not matches:
        return None
    for m in matches:
        if len(m) == 4:
            return int(m)
    for m in matches:
        if len(m) == 2:
            return 2000 + int(m)
    return None


def run_preprocessing(
    raw_excel_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Executa o pipeline completo e salva parquet.
    Retorna caminho do arquivo salvo.
    """
    excel_path = Path(raw_excel_path) if raw_excel_path else RAW_EXCEL
    out_path = Path(output_path) if output_path else REFINED_PARQUET

    sheets = read_raw_excel(excel_path)
    df_all = concat_sheets_with_year(sheets)
    df_all = df_all.sort_values(by=["RA", "ANO"]).reset_index(drop=True)

    df_base = build_features_base_raw(df_all)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_base.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    return out_path


if __name__ == "__main__":
    saved_path = run_preprocessing()
    print(f"Arquivo salvo em {saved_path}")
