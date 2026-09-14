"""Validacion del modelo: validacion cruzada estratificada y diagnostico de over/underfitting.

Compara el desempeno en train, validacion (promedio de los folds) y test. Si el modelo
sobreajusta de forma severa o no supera a la heuristica clinica de la POC, se lanza un
error para que el training pipeline no guarde un modelo que no sirve.
"""

from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from pipelines.config import SEMILLA

METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRICA_PRINCIPAL = "recall"
N_FOLDS = 5
# Brecha de recall train - validacion a partir de la cual se considera overfitting severo
MAX_BRECHA_TRAIN_VALIDACION = 0.2
# Recall de la heuristica clinica del notebook baseline: un modelo peor no aporta nada
MIN_RECALL_VALIDACION = 0.63


class ErrorValidacionModeloError(ValueError):
    """El modelo no pasa la validacion."""


def validar_modelo(
    modelo: BaseEstimator,
    x_train: Any,
    y_train: Any,
    metricas_test: dict[str, float],
) -> dict[str, Any]:
    """Valida con StratifiedKFold y compara train / validacion / test.

    Devuelve el reporte de validacion; lanza ErrorValidacionModeloError si el diagnostico
    es overfitting o underfitting.
    """
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    puntajes = cross_validate(
        clone(modelo), x_train, y_train, cv=folds, scoring=METRICAS, return_train_score=True
    )

    validacion_cruzada = {
        metrica: {
            "train": float(puntajes[f"train_{metrica}"].mean()),
            "validacion": float(puntajes[f"test_{metrica}"].mean()),
            "validacion_std": float(puntajes[f"test_{metrica}"].std()),
        }
        for metrica in METRICAS
    }

    principal = validacion_cruzada[METRICA_PRINCIPAL]
    brecha_train_validacion = principal["train"] - principal["validacion"]
    if brecha_train_validacion > MAX_BRECHA_TRAIN_VALIDACION:
        diagnostico = "overfitting"
    elif principal["validacion"] < MIN_RECALL_VALIDACION:
        diagnostico = "underfitting"
    else:
        diagnostico = "adecuado"

    reporte = {
        "n_folds": N_FOLDS,
        "validacion_cruzada": validacion_cruzada,
        "test": metricas_test,
        "brecha_train_validacion": brecha_train_validacion,
        "brecha_validacion_test": principal["validacion"] - metricas_test[METRICA_PRINCIPAL],
        "diagnostico": diagnostico,
    }

    if diagnostico != "adecuado":
        raise ErrorValidacionModeloError(
            f"Diagnostico: {diagnostico}. {METRICA_PRINCIPAL} train={principal['train']:.3f}, "
            f"validacion={principal['validacion']:.3f}, brecha={brecha_train_validacion:.3f}"
        )
    return reporte
