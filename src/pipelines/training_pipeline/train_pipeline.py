"""Training pipeline (arquitectura FTI): features -> modelo entrenado, evaluado y guardado.

Reconstruye el modelo seleccionado en la POC: el preprocesamiento del notebook de feature
engineering y el Random Forest optimizado del notebook de seleccion del mejor modelo.

Ejecutar desde la raiz del repositorio (despues del feature pipeline):
    uv run python src/pipelines/training_pipeline/train_pipeline.py
"""

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PowerTransformer, RobustScaler

# Permite ejecutarlo como script: agrega src/ al path para importar el paquete pipelines
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.config import (
    PROPORCION_TEST,
    RUTA_FEATURES,
    RUTA_MODELO,
    SEMILLA,
    TARGET,
)
from pipelines.training_pipeline.model_validation import METRICA_PRINCIPAL, validar_modelo
from pipelines.training_pipeline.split_validation import validate_train_test_split

N_BINS_EDAD = 4


def dividir_datos(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Particion train / test estratificada por el target."""
    x = features.drop(columns=TARGET)
    y = features[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=PROPORCION_TEST, random_state=SEMILLA, stratify=y
    )
    return x_train, x_test, y_train, y_test


def construir_modelo() -> Pipeline:
    """Pipeline completo: preprocesamiento por grupo de variables + Random Forest."""
    imputar_mediana = ("imputar", SimpleImputer(strategy="median"))
    imputar_moda = ("imputar", SimpleImputer(strategy="most_frequent"))

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "simetricas",
                Pipeline([imputar_mediana, ("escalar", RobustScaler())]),
                ["age", "rest_bp", "max_hr"],
            ),
            (
                "sesgadas",
                Pipeline([imputar_mediana, ("normalizar", PowerTransformer())]),
                ["chol", "old_peak"],
            ),
            (
                "discretas",
                Pipeline([imputar_mediana, ("escalar", RobustScaler())]),
                ["ca", "slope"],
            ),
            ("booleanas", Pipeline([imputar_moda]), ["fbs", "exang"]),
            (
                "categoricas",
                Pipeline(
                    [
                        imputar_moda,
                        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["sex", "chest_pain", "rest_ecg", "thal"],
            ),
            (
                "edad_grupos",
                Pipeline(
                    [
                        imputar_mediana,
                        (
                            "discretizar",
                            KBinsDiscretizer(n_bins=N_BINS_EDAD, encode="onehot-dense"),
                        ),
                    ]
                ),
                ["age"],
            ),
        ],
        remainder="drop",
    )

    # Hiperparametros elegidos con RandomizedSearchCV en la POC
    clasificador = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=SEMILLA,
    )
    return Pipeline([("preprocesar", preprocesador), ("clasificar", clasificador)])


def evaluar(modelo: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Metricas de clasificacion; recall es la principal (un falso negativo es lo mas caro)."""
    y_pred = modelo.predict(x)
    y_proba = modelo.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


def ejecutar(
    ruta_features: Path = RUTA_FEATURES, ruta_modelo: Path = RUTA_MODELO
) -> dict[str, Any]:
    """Entrena, evalua en test y guarda el modelo junto a sus metadatos."""
    x_train, x_test, y_train, y_test = dividir_datos(pd.read_parquet(ruta_features))
    # Si hay fuga de datos o la particion no es representativa se detiene antes de entrenar
    columnas_con_drift = validate_train_test_split(x_train, x_test, y_train, y_test)

    modelo = construir_modelo()
    modelo.fit(x_train, y_train)
    metricas_test = evaluar(modelo, x_test, y_test)
    # Con overfitting o underfitting severo se lanza el error y el modelo no se guarda
    validacion = validar_modelo(construir_modelo(), x_train, y_train, metricas_test)

    metadatos = {
        "modelo": "Random Forest",
        "metrica_principal": METRICA_PRINCIPAL,
        "resultados_test": metricas_test,
        "validacion": validacion,
        "semilla": SEMILLA,
        "columnas_entrada": list(x_train.columns),
        "columnas_con_drift": columnas_con_drift,
    }

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo)
    joblib.dump(metadatos, ruta_modelo.with_name(f"{ruta_modelo.stem}_metadatos.joblib"))
    return metadatos


if __name__ == "__main__":
    resultado = ejecutar()
    print(f"Modelo guardado en {RUTA_MODELO}")
    print(f"Columnas con drift individual entre train y test: {resultado['columnas_con_drift']}")
    print(f"Diagnostico de la validacion del modelo: {resultado['validacion']['diagnostico']}")
    for metrica, valor in resultado["resultados_test"].items():
        print(f"  {metrica:<10}: {valor:.4f}")
