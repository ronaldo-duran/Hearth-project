"""Datos sinteticos compartidos por las pruebas de los pipelines."""

import pandas as pd
import pytest


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
