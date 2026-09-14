"""Configuracion compartida por los pipelines FTI: rutas y dominio valido de los datos.

Los dominios salen de data/01_raw/datos_corazon_Info.txt y del notebook de exploracion
inicial (notebooks/2-exploration).
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUTA_RAW = RAIZ / "data" / "01_raw" / "corazon.csv"
RUTA_FEATURES = RAIZ / "data" / "03_primary" / "corazon_limpio.parquet"
RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
RUTA_PREDICCIONES = RAIZ / "data" / "07_model_output" / "predicciones.csv"

TARGET = "disease"
SEMILLA = 42
PROPORCION_TEST = 0.2

# Umbral por debajo de 0.5 segun el analisis del issue #14: en screening cardiaco un falso
# negativo (dar por sano a un enfermo) cuesta mucho mas que un falso positivo
UMBRAL_DECISION = 0.411

CATEGORIAS_VALIDAS = {
    "sex": ["Female", "Male"],
    "chest_pain": ["asymptomatic", "nonanginal", "nontypical", "typical"],
    "rest_ecg": ["ST-T wave abnormality", "left ventricular hypertrophy", "normal"],
    "thal": ["fixed", "normal", "reversable"],
}

# Rangos de plausibilidad clinica para las variables continuas
RANGOS_NUMERICOS = {
    "age": (0, 120),
    "rest_bp": (50, 250),
    "chol": (100, 600),
    "max_hr": (60, 220),
    "old_peak": (0.0, 10.0),
}

# Variables numericas con un conjunto cerrado de valores admitidos
VALORES_DISCRETOS = {
    "fbs": [0, 1],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    TARGET: [0, 1],
}

# Columnas que recibe el modelo, en el orden del dataset original
COLUMNAS_ENTRADA = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "fbs",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
]
