"""Inference pipeline (arquitectura FTI): datos nuevos -> predicciones del modelo entrenado.

A los datos nuevos se les aplica el mismo tipado del feature pipeline (tipar_datos) y luego
el pipeline completo que guardo el training pipeline (preprocesamiento + modelo), asi que no
hay ninguna transformacion replicada a mano que se pueda desincronizar del entrenamiento.

Ejecutar desde la raiz del repositorio:
    uv run python src/pipelines/inference_pipeline/inference_pipeline.py <entrada.csv> [salida.csv]
"""

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Permite ejecutarlo como script: agrega src/ al path para importar el paquete pipelines
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.config import COLUMNAS_ENTRADA, RUTA_MODELO, RUTA_PREDICCIONES, UMBRAL_DECISION
from pipelines.feature_pipeline.feature_pipeline import tipar_datos


class ErrorDatosEntradaError(ValueError):
    """Los datos nuevos no tienen el formato que espera el modelo."""


def cargar_modelo(ruta_modelo: Path = RUTA_MODELO) -> Any:
    """Carga el pipeline completo (preprocesamiento + modelo) guardado por el training pipeline."""
    return joblib.load(ruta_modelo)


def predecir(modelo: Any, nuevos: pd.DataFrame, umbral: float = UMBRAL_DECISION) -> pd.DataFrame:
    """Devuelve los registros originales con la probabilidad y la prediccion del modelo.

    Los valores fuera de dominio se vuelven nulos y el modelo los imputa; la columna
    valores_imputados indica cuantos hubo en cada registro para no ocultarlo.
    """
    faltantes = [col for col in COLUMNAS_ENTRADA if col not in nuevos.columns]
    if faltantes:
        raise ErrorDatosEntradaError(f"Faltan columnas requeridas: {faltantes}")
    if nuevos.empty:
        raise ErrorDatosEntradaError("No hay registros para predecir")

    datos = tipar_datos(nuevos[COLUMNAS_ENTRADA])
    probabilidad = modelo.predict_proba(datos)[:, 1]

    resultado = nuevos.copy()
    resultado["valores_imputados"] = datos.isna().sum(axis=1).to_numpy()
    resultado["probabilidad_enfermedad"] = probabilidad.round(4)
    resultado["prediccion"] = (probabilidad >= umbral).astype(int)
    return resultado


def ejecutar(
    ruta_entrada: Path,
    ruta_salida: Path = RUTA_PREDICCIONES,
    ruta_modelo: Path = RUTA_MODELO,
) -> pd.DataFrame:
    """Lee los datos nuevos, genera las predicciones y las persiste en CSV."""
    predicciones = predecir(cargar_modelo(ruta_modelo), pd.read_csv(ruta_entrada))

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(ruta_salida, index=False)
    return predicciones


if __name__ == "__main__":
    if len(sys.argv) < 2:  # noqa: PLR2004
        sys.exit(__doc__)
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else RUTA_PREDICCIONES  # noqa: PLR2004
    resultado = ejecutar(Path(sys.argv[1]), salida)
    print(f"{len(resultado)} predicciones guardadas en {salida}")
    print(f"Pacientes con riesgo detectado: {resultado['prediccion'].sum()}")
