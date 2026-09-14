from pathlib import Path

import pandas as pd

from pipelines.config import COLUMNAS_ENTRADA
from pipelines.feature_pipeline.feature_pipeline import ejecutar, limpiar_datos, tipar_datos

# 8 filas crudas - 1 duplicada - 1 sin target - 2 con etiquetas contradictorias
FILAS_LIMPIAS_ESPERADAS = 4


def test_tipar_datos_convierte_tipos_y_unifica_nulos(datos_crudos: pd.DataFrame) -> None:
    datos = tipar_datos(datos_crudos)

    assert len(datos) == len(datos_crudos), "El tipado no debe eliminar filas"
    assert datos["age"].dtype == float
    assert datos.loc[0, "chest_pain"] == "typical", "Debe quitar espacios sobrantes"
    assert datos.loc[0, "thal"] == "fixed", "Debe quitar espacios sobrantes"
    assert pd.isna(datos.loc[3, "rest_bp"]), "Texto en columna numerica debe ser nulo"
    assert pd.isna(datos.loc[4, "chol"]), "Valor fuera de rango debe ser nulo"
    assert pd.isna(datos.loc[4, "thal"]), "Categoria inexistente debe ser nula"


def test_tipar_datos_no_exige_target(datos_crudos: pd.DataFrame) -> None:
    datos = tipar_datos(datos_crudos.drop(columns="disease"))

    assert list(datos.columns) == COLUMNAS_ENTRADA


def test_limpiar_datos_elimina_registros_invalidos(datos_crudos: pd.DataFrame) -> None:
    limpio = limpiar_datos(tipar_datos(datos_crudos))

    assert len(limpio) == FILAS_LIMPIAS_ESPERADAS
    assert not limpio.duplicated().any()
    assert limpio["disease"].notna().all()
    assert limpio["disease"].dtype == int


def test_ejecutar_persiste_features(datos_crudos: pd.DataFrame, tmp_path: Path) -> None:
    ruta_raw = tmp_path / "corazon.csv"
    ruta_salida = tmp_path / "salida" / "features.parquet"
    datos_crudos.to_csv(ruta_raw, index=False)

    features = ejecutar(ruta_raw, ruta_salida)

    assert ruta_salida.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(ruta_salida), features, check_dtype=False)
