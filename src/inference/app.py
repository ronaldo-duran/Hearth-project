"""Demo del modelo de prediccion de enfermedad cardiaca con Streamlit.

Dos modos en la misma interfaz:
- Prediccion individual (issue #29): formulario con los datos de un examen medico.
- Prediccion por lotes (issue #30): un CSV con varios pacientes -> tabla y descarga.

Ambos usan el inference pipeline (src/pipelines/inference_pipeline), asi que la app aplica
exactamente el mismo tipado, el mismo modelo y el mismo umbral que el pipeline FTI.

Ejecutar con:
    uv run streamlit run src/inference/app.py
"""

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# Agrega src/ al path para importar el paquete pipelines
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.config import CATEGORIAS_VALIDAS, RUTA_MODELO, UMBRAL_DECISION
from pipelines.inference_pipeline.inference_pipeline import (
    ErrorDatosEntradaError,
    cargar_modelo,
    predecir,
)

RUTA_EJEMPLO = Path(__file__).parent / "ejemplos" / "pacientes_ejemplo.csv"

ETIQUETAS_DOLOR = {
    "typical": "Angina típica",
    "nontypical": "Angina atípica",
    "nonanginal": "Dolor no anginoso",
    "asymptomatic": "Asintomático",
}

ETIQUETAS_THAL = {
    "normal": "Normal",
    "fixed": "Defecto fijo",
    "reversable": "Defecto reversible",
}

ETIQUETAS_PENDIENTE = {1: "1 - Ascendente", 2: "2 - Plana", 3: "3 - Descendente"}


@st.cache_resource
def cargar_modelo_y_metadatos() -> tuple[Any, dict[str, Any]]:
    """Carga el pipeline completo (preprocesamiento + modelo) una sola vez."""
    metadatos = joblib.load(RUTA_MODELO.with_name(f"{RUTA_MODELO.stem}_metadatos.joblib"))
    return cargar_modelo(), metadatos


def construir_formulario() -> pd.DataFrame:
    """Dibuja el formulario y devuelve los datos del paciente como DataFrame."""
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("##### Datos del paciente")
        age = st.slider("Edad (años)", 20, 100, 55)
        sex = st.radio("Sexo", CATEGORIAS_VALIDAS["sex"], horizontal=True)
        chest_pain = st.selectbox(
            "Tipo de dolor torácico",
            CATEGORIAS_VALIDAS["chest_pain"],
            index=0,
            format_func=lambda v: ETIQUETAS_DOLOR[v],
        )
        rest_bp = st.slider("Presión arterial en reposo (mm Hg)", 80, 220, 130)
        chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 245)
        fbs = st.checkbox("Glucosa en ayunas > 120 mg/dl")
        rest_ecg = st.selectbox(
            "Electrocardiograma en reposo", CATEGORIAS_VALIDAS["rest_ecg"], index=2
        )

    with col_der:
        st.markdown("##### Prueba de esfuerzo y perfusión")
        max_hr = st.slider("Frecuencia cardíaca máxima (lpm)", 60, 220, 150)
        exang = st.checkbox("Angina inducida por el ejercicio")
        old_peak = st.slider("Depresión del segmento ST (old_peak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST",
            [1, 2, 3],
            index=1,
            format_func=lambda v: ETIQUETAS_PENDIENTE[v],
        )
        ca = st.selectbox("Vasos principales coloreados por fluoroscopia", [0, 1, 2, 3], index=0)
        thal = st.selectbox(
            "Gammagrafía con talio",
            CATEGORIAS_VALIDAS["thal"],
            index=1,
            format_func=lambda v: ETIQUETAS_THAL[v],
        )

    return pd.DataFrame(
        [
            {
                "age": float(age),
                "sex": sex,
                "chest_pain": chest_pain,
                "rest_bp": float(rest_bp),
                "chol": float(chol),
                "fbs": float(fbs),
                "rest_ecg": rest_ecg,
                "max_hr": float(max_hr),
                "exang": float(exang),
                "old_peak": float(old_peak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": thal,
            }
        ]
    )


def mostrar_resultado(probabilidad: float) -> None:
    """Muestra la probabilidad de enfermedad y la decision segun el umbral."""
    hay_enfermedad = probabilidad >= UMBRAL_DECISION

    st.markdown("### Resultado")
    columna_metrica, columna_mensaje = st.columns([1, 2])

    with columna_metrica:
        st.metric("Probabilidad de enfermedad", f"{probabilidad:.1%}")

    with columna_mensaje:
        if hay_enfermedad:
            st.error("**Riesgo detectado** — se recomienda valoración cardiológica.")
        else:
            st.success("**Sin indicios de enfermedad** según el modelo.")

    st.progress(min(float(probabilidad), 1.0))
    st.caption(
        f"Umbral de decisión: {UMBRAL_DECISION:.3f}. "
        "Se usa un umbral por debajo de 0.5 a propósito: en screening cardíaco es "
        "preferible enviar a un sano a un examen adicional que dar por sano a un enfermo."
    )


def pestana_individual(modelo: Any) -> None:
    """Prediccion de un paciente a partir del formulario."""
    datos_paciente = construir_formulario()

    st.divider()
    if st.button("Predecir", type="primary", width="stretch"):
        resultado = predecir(modelo, datos_paciente)
        mostrar_resultado(float(resultado["probabilidad_enfermedad"].iloc[0]))

        with st.expander("Ver los datos enviados al modelo"):
            st.dataframe(datos_paciente, width="stretch")


def pestana_lotes(modelo: Any) -> None:
    """Prediccion de varios pacientes a partir de un archivo CSV."""
    st.markdown(
        "Suba un archivo **CSV** con un paciente por fila y las 13 columnas del examen: "
        "`age, sex, chest_pain, rest_bp, chol, fbs, rest_ecg, max_hr, exang, old_peak, "
        "slope, ca, thal`. Las columnas adicionales se conservan en el resultado."
    )
    st.download_button(
        "Descargar CSV de ejemplo",
        RUTA_EJEMPLO.read_bytes(),
        file_name="pacientes_ejemplo.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Archivo CSV con los pacientes", type="csv")
    if archivo is None:
        return

    try:
        predicciones = predecir(modelo, pd.read_csv(archivo))
    except (ErrorDatosEntradaError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"No se pudo procesar el archivo: {error}")
        return

    col_total, col_riesgo, col_imputados = st.columns(3)
    col_total.metric("Pacientes", len(predicciones))
    col_riesgo.metric("Con riesgo detectado", int(predicciones["prediccion"].sum()))
    col_imputados.metric(
        "Con valores inválidos imputados", int((predicciones["valores_imputados"] > 0).sum())
    )

    st.dataframe(
        predicciones,
        width="stretch",
        column_config={
            "probabilidad_enfermedad": st.column_config.ProgressColumn(
                "probabilidad_enfermedad", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )
    st.caption(
        f"prediccion = 1 si la probabilidad es mayor o igual al umbral ({UMBRAL_DECISION:.3f}). "
        "valores_imputados cuenta los datos faltantes o fuera de dominio que el modelo tuvo "
        "que imputar en cada paciente."
    )
    st.download_button(
        "Descargar predicciones (CSV)",
        predicciones.to_csv(index=False).encode("utf-8"),
        file_name="predicciones.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    """Punto de entrada de la aplicacion Streamlit."""
    st.set_page_config(
        page_title="Predicción de enfermedad cardíaca", page_icon="🫀", layout="wide"
    )

    modelo, metadatos = cargar_modelo_y_metadatos()

    st.title("🫀 Predicción de enfermedad cardíaca")
    st.markdown(
        "Demo del modelo entrenado en el proyecto **Hearth-project**. "
        "Prediga un paciente con el formulario o varios a la vez con un archivo CSV."
    )

    st.warning(
        "**Herramienta académica.** No es un dispositivo médico ni sustituye el criterio "
        "de un profesional de la salud. Entrenada con 480 pacientes del estudio Cleveland "
        "(UCI), de los años 80.",
        icon="⚠️",
    )

    with st.sidebar:
        st.header("Modelo")
        st.write(f"**Algoritmo:** {metadatos['modelo']}")
        st.write(f"**Métrica principal:** {metadatos['metrica_principal']}")
        st.markdown("**Desempeño en test:**")
        resultados = pd.Series(metadatos["resultados_test"]).round(3).to_frame("valor")
        st.dataframe(resultados, width="stretch")
        st.caption(
            "El modelo se seleccionó comparando 5 familias con validación cruzada y una "
            "prueba estadística (t-test corregido de Nadeau-Bengio)."
        )

    individual, lotes = st.tabs(["🩺 Predicción individual", "📄 Predicción por lotes (CSV)"])
    with individual:
        pestana_individual(modelo)
    with lotes:
        pestana_lotes(modelo)


if __name__ == "__main__":
    main()
