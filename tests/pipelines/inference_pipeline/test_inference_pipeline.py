from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from pipelines.config import COLUMNAS_ENTRADA, TARGET
from pipelines.inference_pipeline.inference_pipeline import (
    ErrorDatosEntradaError,
    ejecutar,
    predecir,
)


class ModeloFalso:
    """Modelo dummy que usa la edad / 100 como probabilidad y guarda lo que recibio."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.recibido = x
        probabilidad = x["age"].fillna(0).to_numpy() / 100
        return np.column_stack([1 - probabilidad, probabilidad])


def test_predecir_aplica_umbral(datos_crudos: pd.DataFrame) -> None:
    nuevos = datos_crudos.drop(columns=TARGET).assign(age=[30, 40, 41, 50, 60, 70, 80, 90])

    resultado = predecir(ModeloFalso(), nuevos, umbral=0.41)

    assert resultado["prediccion"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert resultado["probabilidad_enfermedad"].iloc[0] == pytest.approx(0.30)


def test_predecir_aplica_el_tipado_del_feature_pipeline(datos_crudos: pd.DataFrame) -> None:
    modelo = ModeloFalso()

    resultado = predecir(modelo, datos_crudos)

    assert list(modelo.recibido.columns) == COLUMNAS_ENTRADA, "El target no llega al modelo"
    assert modelo.recibido.loc[0, "chest_pain"] == "typical", "Debe quitar espacios"
    assert pd.isna(modelo.recibido.loc[3, "rest_bp"]), "Valor corrupto debe llegar como nulo"
    assert resultado.loc[3, "valores_imputados"] == 1
    assert resultado.loc[4, "valores_imputados"] == 2  # noqa: PLR2004
    assert TARGET in resultado.columns, "Conserva las columnas originales"


def test_predecir_rechaza_datos_invalidos(datos_crudos: pd.DataFrame) -> None:
    with pytest.raises(ErrorDatosEntradaError, match="Faltan columnas"):
        predecir(ModeloFalso(), datos_crudos.drop(columns=["age", "thal"]))

    with pytest.raises(ErrorDatosEntradaError, match="No hay registros"):
        predecir(ModeloFalso(), datos_crudos.head(0))


def test_ejecutar_con_modelo_dummy_guardado(
    features_sinteticas: pd.DataFrame, datos_crudos: pd.DataFrame, tmp_path: Path
) -> None:
    x = features_sinteticas[COLUMNAS_ENTRADA]
    modelo = DummyClassifier(strategy="prior").fit(x, features_sinteticas[TARGET])
    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(modelo, ruta_modelo)
    ruta_entrada = tmp_path / "nuevos.csv"
    datos_crudos.drop(columns=TARGET).to_csv(ruta_entrada, index=False)
    ruta_salida = tmp_path / "salida" / "predicciones.csv"

    predicciones = ejecutar(ruta_entrada, ruta_salida, ruta_modelo)

    assert ruta_salida.exists()
    assert len(pd.read_csv(ruta_salida)) == len(datos_crudos)
    assert predicciones["probabilidad_enfermedad"].between(0, 1).all()
