from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.data_validation import ErrorValidacionDatosError, validar_datos
from pipelines.feature_pipeline.feature_pipeline import ejecutar, limpiar_datos, tipar_datos


@pytest.fixture
def features(datos_crudos: pd.DataFrame) -> pd.DataFrame:
    return limpiar_datos(tipar_datos(datos_crudos))


def test_datos_validos_pasan(features: pd.DataFrame) -> None:
    validar_datos(features, max_nulos=0.5)


@pytest.mark.parametrize(
    ("columna", "valor", "mensaje"),
    [
        ("age", 150.0, "fuera de rango"),
        ("ca", 7.0, "fuera de rango"),
        ("thal", "otro", "categorias no validas"),
        ("chol", "alto", "se esperaba numerico"),
    ],
)
def test_detecta_valores_invalidos(
    features: pd.DataFrame, columna: str, valor: object, mensaje: str
) -> None:
    if isinstance(valor, str):
        features[columna] = features[columna].astype(object)
    features.loc[0, columna] = valor

    with pytest.raises(ErrorValidacionDatosError, match=mensaje):
        validar_datos(features, max_nulos=0.5)


def test_detecta_exceso_de_nulos(features: pd.DataFrame) -> None:
    features.loc[:2, "chol"] = None

    with pytest.raises(ErrorValidacionDatosError, match="de nulos"):
        validar_datos(features, max_nulos=0.5)


def test_detecta_duplicados_y_columnas_faltantes(features: pd.DataFrame) -> None:
    with pytest.raises(ErrorValidacionDatosError, match="duplicados"):
        validar_datos(pd.concat([features, features.head(1)]), max_nulos=0.5)

    with pytest.raises(ErrorValidacionDatosError, match="Faltan columnas"):
        validar_datos(features.drop(columns="thal"))


def test_pipeline_no_persiste_si_la_validacion_falla(
    datos_crudos: pd.DataFrame, tmp_path: Path
) -> None:
    ruta_raw = tmp_path / "corazon.csv"
    ruta_salida = tmp_path / "features.parquet"
    # Con 4 filas limpias, la columna con un unico valor valido supera el 30% de nulos
    datos_crudos.assign(chol="?").to_csv(ruta_raw, index=False)

    with pytest.raises(ErrorValidacionDatosError):
        ejecutar(ruta_raw, ruta_salida)

    assert not ruta_salida.exists()
