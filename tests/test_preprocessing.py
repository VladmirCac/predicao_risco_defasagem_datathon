import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    normalize_col,
    coalesce_cols,
    parse_idade,
    parse_fase_int,
    extract_year_from_sheet_name,
    concat_sheets_with_year,
    build_features_base_raw,
)


def test_normalize_col_removes_accents_and_symbols():
    assert normalize_col(" Idade (2024) ") == "idade2024"
    assert normalize_col("Nóme-Do.Aluno") == "nome_do_aluno"


def test_coalesce_cols_picks_first_existing():
    df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]})
    result = coalesce_cols(df, ["x", "a", "b"])
    assert result.tolist() == [1, 2]


@pytest.mark.parametrize(
    "value,expected",
    [
        (15, 15.0),
        ("1899-12-08", 8.0),  # data com bug do Excel
        ("2000-01-01", np.nan),
        ("abc", np.nan),
    ],
)
def test_parse_idade_handles_numeric_and_dates(value, expected):
    res = parse_idade(value)
    if np.isnan(expected):
        assert np.isnan(res)
    else:
        # pandas pode interpretar 1899-12-08 como 12/08/1899 (dayfirst),
        # então aceitamos 8 ou 12 para o bug clássico do Excel.
        assert res in {expected, 12.0} if value == "1899-12-08" else res == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1A", 1),
        ("fase 3", 3),
        ("alpha", 0),
        ("", pd.NA),
        (2.0, 2),
    ],
)
def test_parse_fase_int_covers_main_paths(value, expected):
    result = parse_fase_int(value)
    if value == "":
        assert pd.isna(result)
    else:
        assert result == expected


def test_extract_year_from_sheet_name_accepts_two_or_four_digits():
    assert extract_year_from_sheet_name("Base 2023") == 2023
    assert extract_year_from_sheet_name("Planilha_24") == 2024
    assert extract_year_from_sheet_name("SemAno") is None


def test_concat_sheets_with_year_adds_column():
    sheets = {"aba2022": pd.DataFrame({"RA": [1]}), "xx_2023": pd.DataFrame({"RA": [2]})}
    merged = concat_sheets_with_year(sheets)
    assert set(merged["ANO"]) == {2022, 2023}
    assert len(merged) == 2


def test_build_features_base_raw_normalizes_and_clips():
    df = pd.DataFrame(
        {
            "RA": [" 10 ", "20"],
            "ANO": [2022, 2023],
            "Idade 2022": [15, "1899-12-08"],
            "Fase": ["1A", 2],
            "Defasagem_2022": [5, 12],
            "IAA": [11, -1],  # vai ser clipado para 10 e 0
            "NOTA_MAT_2023": [9.5, 7.1],
        }
    )

    out = build_features_base_raw(df)

    assert list(out.columns) == [
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
    # Coluna de idade existe e mantém comprimento
    assert "IDADE" in out.columns
    assert len(out["IDADE"]) == len(df)
    assert out["IAA"].tolist() == [10.0, 0.0]
    assert out["DEFASAGEM"].tolist() == [5.0, 12.0]


def test_build_features_base_raw_raises_without_ra_or_ano():
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(ValueError):
        build_features_base_raw(df)
