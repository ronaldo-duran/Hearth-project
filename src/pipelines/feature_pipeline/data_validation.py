"""Reglas de validacion e integridad de las features antes de persistirlas.

Se implementa con pandas: las reglas son pocas y ya estan definidas en pipelines.config,
asi que no se justifica agregar Great Expectations o Pandera como dependencia.
"""

import pandas as pd

from pipelines.config import (
    CATEGORIAS_VALIDAS,
    COLUMNAS_ENTRADA,
    RANGOS_NUMERICOS,
    TARGET,
    VALORES_DISCRETOS,
)

# El dataset real llega hasta 24% de nulos (rest_ecg); por encima de 30% la columna
# deja de ser confiable para imputar
MAX_PROPORCION_NULOS = 0.3


class ErrorValidacionDatosError(ValueError):
    """Los datos no cumplen las reglas de validacion."""


def _errores_columna(serie: pd.Series, max_nulos: float) -> list[str]:
    """Revisa tipo, dominio y proporcion de nulos de una columna."""
    col = str(serie.name)
    presentes = serie.dropna()
    errores = []

    if col in CATEGORIAS_VALIDAS:
        if not pd.api.types.is_string_dtype(serie):
            errores.append(f"{col}: tipo {serie.dtype}, se esperaba texto")
        invalidas = sorted(set(presentes) - set(CATEGORIAS_VALIDAS[col]))
        if invalidas:
            errores.append(f"{col}: categorias no validas {invalidas}")
    elif not pd.api.types.is_numeric_dtype(serie):
        errores.append(f"{col}: tipo {serie.dtype}, se esperaba numerico")
    else:
        if col in VALORES_DISCRETOS:
            fuera = presentes[~presentes.isin(VALORES_DISCRETOS[col])]
        else:
            fuera = presentes[~presentes.between(*RANGOS_NUMERICOS[col])]
        if not fuera.empty:
            errores.append(f"{col}: {len(fuera)} valores fuera de rango")

    proporcion_nulos = serie.isna().mean()
    if proporcion_nulos > max_nulos:
        errores.append(f"{col}: {proporcion_nulos:.1%} de nulos, maximo {max_nulos:.0%}")
    return errores


def validar_datos(datos: pd.DataFrame, max_nulos: float = MAX_PROPORCION_NULOS) -> None:
    """Valida tipos, rangos, nulos, categorias y unicidad; lanza error si algo falla.

    El dataset no tiene columnas de fecha, asi que no aplica la validacion de formatos.
    """
    esperadas = [*COLUMNAS_ENTRADA, TARGET]
    faltantes = sorted(set(esperadas) - set(datos.columns))
    if faltantes:
        raise ErrorValidacionDatosError(f"Faltan columnas: {faltantes}")

    errores = [e for col in esperadas for e in _errores_columna(datos[col], max_nulos)]

    if datos[TARGET].isna().any():
        errores.append(f"{TARGET}: el target no puede tener nulos")
    if datos.duplicated().any():
        errores.append(f"{datos.duplicated().sum()} registros duplicados")

    if errores:
        raise ErrorValidacionDatosError("Validacion de datos fallida:\n- " + "\n- ".join(errores))
