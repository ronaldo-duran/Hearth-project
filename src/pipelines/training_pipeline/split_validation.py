"""Verificaciones de la particion train / test: fuga de datos y representatividad.

- Fuga de datos: indices compartidos y examenes identicos presentes en ambos conjuntos.
- Representatividad: proporcion del target y drift de las features con Evidently.

La guia del curso usa DeepChecks, pero su version actual no importa con scikit-learn 1.9
(usa scorers que ya no existen), asi que la comparacion de distribuciones se hace con
Evidently, la otra opcion que admite el enunciado.
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Con estratificacion la diferencia deberia ser casi nula; 5 puntos ya indica un problema
MAX_DIFERENCIA_TARGET = 0.05
# Mismo criterio de "dataset drift" que usa Evidently por defecto
MAX_PROPORCION_COLUMNAS_CON_DRIFT = 0.5


class ErrorParticionError(ValueError):
    """La particion train / test no es valida."""


def columnas_con_drift(x_train: pd.DataFrame, x_test: pd.DataFrame) -> list[str]:
    """Columnas cuya distribucion en test difiere de la de train segun Evidently."""
    snapshot = Report([DataDriftPreset()]).run(reference_data=x_train, current_data=x_test)

    con_drift = []
    for metrica in snapshot.dict()["metrics"]:
        config = metrica["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        valor, umbral = metrica["value"], config["threshold"]
        # Las pruebas que devuelven p-valor indican drift por debajo del umbral; las de
        # distancia (Evidently las usa con muestras grandes), por encima
        hay_drift = valor < umbral if "p_value" in config["method"] else valor >= umbral
        if hay_drift:
            con_drift.append(config["column"])
    return con_drift


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[str]:
    """Valida la particion; lanza ErrorParticionError si hay fuga o no es representativa.

    Devuelve las columnas con drift individual, que se reportan pero no bloquean mientras
    no superen la proporcion maxima.
    """
    errores = []

    if list(x_train.columns) != list(x_test.columns):
        errores.append("train y test no tienen las mismas columnas")

    indices_compartidos = x_train.index.intersection(x_test.index)
    if not indices_compartidos.empty:
        errores.append(f"fuga de datos: {len(indices_compartidos)} indices en train y test")

    examenes_repetidos = x_test.merge(x_train.drop_duplicates(), how="inner")
    if not examenes_repetidos.empty:
        errores.append(f"fuga de datos: {len(examenes_repetidos)} examenes identicos en ambos")

    diferencia_target = abs(y_train.mean() - y_test.mean())
    if diferencia_target > MAX_DIFERENCIA_TARGET:
        errores.append(f"proporcion del target difiere {diferencia_target:.1%} entre train y test")

    con_drift = columnas_con_drift(x_train, x_test)
    if len(con_drift) / x_train.shape[1] > MAX_PROPORCION_COLUMNAS_CON_DRIFT:
        errores.append(f"drift en {len(con_drift)} de {x_train.shape[1]} columnas: {con_drift}")

    if errores:
        raise ErrorParticionError("Particion train / test invalida:\n- " + "\n- ".join(errores))
    return con_drift
