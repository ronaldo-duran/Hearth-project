"""Feature pipeline (arquitectura FTI): datos RAW -> features limpias listas para entrenar.

Lleva a un script autonomo lo hecho en los notebooks de exploracion inicial (tipado y
unificacion de nulos) y de feature engineering (limpieza de registros).

Ejecutar desde la raiz del repositorio:
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd

# Permite ejecutarlo como script: agrega src/ al path para importar el paquete pipelines
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.config import (
    CATEGORIAS_VALIDAS,
    COLUMNAS_ENTRADA,
    RANGOS_NUMERICOS,
    RUTA_FEATURES,
    RUTA_RAW,
    TARGET,
    VALORES_DISCRETOS,
)
from pipelines.feature_pipeline.data_validation import validar_datos


def tipar_datos(crudo: pd.DataFrame) -> pd.DataFrame:
    """Convierte cada columna a su tipo y unifica los nulos.

    Los valores fuera de dominio (texto en columnas numericas, categorias inexistentes,
    medidas clinicamente imposibles) se vuelven nulos. Las numericas quedan en float y las
    categoricas en texto, que son los tipos que espera el pipeline de scikit-learn.
    El target es opcional, para poder reutilizar la funcion en inferencia.
    """
    columnas = [c for c in [*COLUMNAS_ENTRADA, TARGET] if c in crudo.columns]
    datos = pd.DataFrame(index=crudo.index)

    for col in columnas:
        texto = crudo[col].astype("str").str.strip()
        if col in CATEGORIAS_VALIDAS:
            datos[col] = texto.where(texto.isin(CATEGORIAS_VALIDAS[col]))
            continue

        numero = pd.to_numeric(texto, errors="coerce")
        if col in VALORES_DISCRETOS:
            valido = numero.isin(VALORES_DISCRETOS[col])
        else:
            minimo, maximo = RANGOS_NUMERICOS[col]
            valido = numero.between(minimo, maximo)
        datos[col] = numero.where(valido).astype(float)

    return datos


def limpiar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, filas sin target y examenes identicos con etiquetas distintas."""
    limpio = datos.drop_duplicates().dropna(subset=[TARGET])

    predictores = [c for c in limpio.columns if c != TARGET]
    etiquetas_distintas = limpio.groupby(predictores, dropna=False)[TARGET].transform("nunique")
    limpio = limpio[etiquetas_distintas == 1].reset_index(drop=True)

    limpio[TARGET] = limpio[TARGET].astype(int)
    return limpio


def ejecutar(ruta_raw: Path = RUTA_RAW, ruta_salida: Path = RUTA_FEATURES) -> pd.DataFrame:
    """Lee los datos RAW, genera las features y las persiste en parquet."""
    features = limpiar_datos(tipar_datos(pd.read_csv(ruta_raw)))
    # Si alguna regla falla se lanza el error antes de escribir: nunca se persisten datos invalidos
    validar_datos(features)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(ruta_salida, index=False)
    return features


if __name__ == "__main__":
    features = ejecutar()
    print(
        f"Features guardadas en {RUTA_FEATURES}: {features.shape[0]} filas x {features.shape[1]} columnas"
    )
