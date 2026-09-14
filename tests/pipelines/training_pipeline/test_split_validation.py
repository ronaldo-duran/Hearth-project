import pandas as pd
import pytest

from pipelines.training_pipeline.split_validation import (
    ErrorParticionError,
    validate_train_test_split,
)
from pipelines.training_pipeline.train_pipeline import dividir_datos


def test_particion_estratificada_es_valida(features_sinteticas: pd.DataFrame) -> None:
    columnas_con_drift = validate_train_test_split(*dividir_datos(features_sinteticas))

    assert isinstance(columnas_con_drift, list)


def test_detecta_indices_compartidos(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)
    x_test.index = x_train.index[: len(x_test)]

    with pytest.raises(ErrorParticionError, match="indices en train y test"):
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_detecta_examenes_identicos(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)
    x_test.iloc[0] = x_train.iloc[0]

    with pytest.raises(ErrorParticionError, match="examenes identicos"):
        validate_train_test_split(x_train, x_test, y_train, y_test)


def test_detecta_target_desbalanceado(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)

    with pytest.raises(ErrorParticionError, match="proporcion del target"):
        validate_train_test_split(x_train, x_test, y_train, y_test.clip(lower=1))


def test_detecta_drift_en_las_features(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)
    # Test con otra poblacion: numericas desplazadas y una sola categoria por columna
    x_test = x_test.assign(
        age=x_test["age"] + 60,
        rest_bp=x_test["rest_bp"] + 100,
        chol=x_test["chol"] + 300,
        max_hr=x_test["max_hr"] + 100,
        old_peak=x_test["old_peak"] + 10,
        sex="Male",
        chest_pain="typical",
        rest_ecg="normal",
        thal="fixed",
        fbs=1.0,
        exang=1.0,
    )

    with pytest.raises(ErrorParticionError, match="drift en"):
        validate_train_test_split(x_train, x_test, y_train, y_test)
