from pathlib import Path

import joblib
import pandas as pd
import pytest

from pipelines.config import PROPORCION_TEST, TARGET
from pipelines.training_pipeline.train_pipeline import (
    construir_modelo,
    dividir_datos,
    ejecutar,
    evaluar,
)


def test_dividir_datos_estratifica(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)

    assert len(x_test) == pytest.approx(len(features_sinteticas) * PROPORCION_TEST, abs=1)
    assert TARGET not in x_train.columns
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)
    assert set(x_train.index).isdisjoint(x_test.index)


def test_modelo_entrena_con_nulos_y_evalua(features_sinteticas: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = dividir_datos(features_sinteticas)

    modelo = construir_modelo().fit(x_train, y_train)
    metricas = evaluar(modelo, x_test, y_test)

    assert set(metricas) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_ejecutar_guarda_modelo_y_metadatos(
    features_sinteticas: pd.DataFrame, tmp_path: Path
) -> None:
    ruta_features = tmp_path / "features.parquet"
    ruta_modelo = tmp_path / "modelos" / "modelo.joblib"
    features_sinteticas.to_parquet(ruta_features, index=False)

    metadatos = ejecutar(ruta_features, ruta_modelo)

    modelo = joblib.load(ruta_modelo)
    assert joblib.load(tmp_path / "modelos" / "modelo_metadatos.joblib") == metadatos
    assert len(modelo.predict(features_sinteticas.drop(columns=TARGET))) == len(features_sinteticas)
