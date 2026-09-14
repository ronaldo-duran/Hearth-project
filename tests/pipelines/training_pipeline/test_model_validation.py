import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from pipelines.training_pipeline.model_validation import (
    METRICAS,
    N_FOLDS,
    ErrorValidacionModeloError,
    validar_modelo,
)
from pipelines.training_pipeline.train_pipeline import construir_modelo, dividir_datos

METRICAS_TEST = dict.fromkeys(METRICAS, 0.9)
COLUMNAS_SIN_NULOS = ["age", "rest_bp", "max_hr", "old_peak"]


def test_modelo_adecuado_devuelve_reporte(features_sinteticas: pd.DataFrame) -> None:
    x_train, _, y_train, _ = dividir_datos(features_sinteticas)

    reporte = validar_modelo(construir_modelo(), x_train, y_train, METRICAS_TEST)

    assert reporte["diagnostico"] == "adecuado"
    assert reporte["n_folds"] == N_FOLDS
    assert set(reporte["validacion_cruzada"]) == set(METRICAS)
    assert set(reporte["validacion_cruzada"]["recall"]) == {"train", "validacion", "validacion_std"}


def test_detecta_overfitting(features_sinteticas: pd.DataFrame) -> None:
    # Un arbol sin limite memoriza etiquetas aleatorias: train perfecto, validacion al azar
    x = features_sinteticas[COLUMNAS_SIN_NULOS]
    y = pd.Series(np.random.default_rng(0).integers(0, 2, len(x)))

    with pytest.raises(ErrorValidacionModeloError, match="overfitting"):
        validar_modelo(DecisionTreeClassifier(random_state=0), x, y, METRICAS_TEST)


def test_detecta_underfitting(features_sinteticas: pd.DataFrame) -> None:
    # Predecir siempre "sano" nunca detecta un enfermo: recall 0 en train y validacion
    x = features_sinteticas[COLUMNAS_SIN_NULOS]
    y = features_sinteticas["disease"]

    with pytest.raises(ErrorValidacionModeloError, match="underfitting"):
        validar_modelo(DummyClassifier(strategy="constant", constant=0), x, y, METRICAS_TEST)
