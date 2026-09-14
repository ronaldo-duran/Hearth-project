"""Datos sinteticos compartidos por las pruebas de los pipelines."""

import numpy as np
import pandas as pd
import pytest

from pipelines.config import (
    CATEGORIAS_VALIDAS,
    COLUMNAS_ENTRADA,
    RANGOS_NUMERICOS,
    SEMILLA,
    TARGET,
    VALORES_DISCRETOS,
)


@pytest.fixture
def datos_crudos() -> pd.DataFrame:
    """Muestra con el formato de data/01_raw/corazon.csv y sus problemas tipicos.

    Incluye espacios sobrantes, valores corruptos, una fila duplicada, un target nulo y dos
    examenes identicos con etiquetas contradictorias.
    """
    filas = [
        [63, "Male", "typical ", 145, 233, 1, "normal", 150, 0, 2.3, 3, 0.0, " fixed", 0],
        [67, "Male", "asymptomatic", 160, 286, 0, "normal", 108, 1, 1.5, 2, 3.0, "normal", 1],
        [67, "Male", "asymptomatic", 160, 286, 0, "normal", 108, 1, 1.5, 2, 3.0, "normal", 1],
        [37, "Female", "nonanginal", "?", 250, 0, "normal", 187, 0, 3.5, 3, 0.0, "normal", 0],
        [41, "Female", "nontypical", 130, 999, 0, "normal", 172, 0, 1.4, 1, 0.0, "abc", 0],
        [56, "Male", "nontypical", 120, 236, 0, "normal", 178, 0, 0.8, 1, 0.0, "normal", None],
        [62, "Female", "asymptomatic", 140, 268, 0, "normal", 160, 0, 3.6, 3, 2.0, "normal", 1],
        [62, "Female", "asymptomatic", 140, 268, 0, "normal", 160, 0, 3.6, 3, 2.0, "normal", 0],
    ]
    columnas = [
        "age",
        "sex",
        "chest_pain",
        "rest_bp",
        "chol",
        "fbs",
        "rest_ecg",
        "max_hr",
        "exang",
        "old_peak",
        "slope",
        "ca",
        "thal",
        "disease",
    ]
    return pd.DataFrame(filas, columns=columnas)


@pytest.fixture
def features_sinteticas() -> pd.DataFrame:
    """Features limpias (salida del feature pipeline) con filas suficientes para entrenar.

    El target depende de old_peak para que el modelo tenga una senal que aprender, y chol
    lleva nulos para ejercitar la imputacion.
    """
    rng = np.random.default_rng(SEMILLA)
    n_filas = 120
    columnas = {}
    for col in COLUMNAS_ENTRADA:
        if col in CATEGORIAS_VALIDAS:
            columnas[col] = rng.choice(CATEGORIAS_VALIDAS[col], n_filas)
        elif col in VALORES_DISCRETOS:
            columnas[col] = rng.choice(VALORES_DISCRETOS[col], n_filas).astype(float)
        else:
            columnas[col] = rng.uniform(*RANGOS_NUMERICOS[col], n_filas).round(1)

    features = pd.DataFrame(columnas)
    features.loc[rng.choice(n_filas, 10, replace=False), "chol"] = np.nan
    features[TARGET] = (features["old_peak"] > 5).astype(int)  # noqa: PLR2004
    return features
