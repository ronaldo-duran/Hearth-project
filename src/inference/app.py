"""Demo funcional del modelo de prediccion de enfermedad cardiaca (issue #15).

Formulario web que recibe los datos de un examen medico en su formato crudo y
devuelve la prediccion del pipeline entrenado (preprocesamiento + modelo).

Ejecutar con:
    uv run streamlit run src/inference/app.py
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# El umbral por defecto de 0.5 se baja a 0.411 segun el analisis del issue #14:
# en screening cardiaco el falso negativo (dar por sano a un enfermo) cuesta
# mucho mas que el falso positivo, asi que se privilegia el recall.
UMBRAL_DECISION = 0.411

CATEGORIAS = {
    "sex": ["Female", "Male"],
    "chest_pain": ["asymptomatic", "nonanginal", "nontypical", "typical"],
    "rest_ecg": ["ST-T wave abnormality", "left ventricular hypertrophy", "normal"],
    "thal": ["fixed", "normal", "reversable"],
}

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
def cargar_modelo() -> tuple[Any, dict[str, Any]]:
    """Carga el pipeline completo (preprocesamiento + modelo) una sola vez."""
    raiz = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "models" / "modelo_final.joblib").exists()
    )
    modelo = joblib.load(raiz / "models" / "modelo_final.joblib")
    metadatos = joblib.load(raiz / "models" / "modelo_final_metadatos.joblib")
    return modelo, metadatos


def construir_formulario() -> pd.DataFrame:
    """Dibuja el formulario y devuelve los datos del paciente como DataFrame."""
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("##### Datos del paciente")
        age = st.slider("Edad (años)", 20, 100, 55)
        sex = st.radio("Sexo", CATEGORIAS["sex"], horizontal=True)
        chest_pain = st.selectbox(
            "Tipo de dolor torácico",
            CATEGORIAS["chest_pain"],
            index=0,
            format_func=lambda v: ETIQUETAS_DOLOR[v],
        )
        rest_bp = st.slider("Presión arterial en reposo (mm Hg)", 80, 220, 130)
        chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 245)
        fbs = st.checkbox("Glucosa en ayunas > 120 mg/dl")
        rest_ecg = st.selectbox("Electrocardiograma en reposo", CATEGORIAS["rest_ecg"], index=2)

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
            CATEGORIAS["thal"],
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


def main() -> None:
    """Punto de entrada de la aplicacion Streamlit."""
    st.set_page_config(
        page_title="Predicción de enfermedad cardíaca", page_icon="🫀", layout="wide"
    )

    modelo, metadatos = cargar_modelo()

    st.title("🫀 Predicción de enfermedad cardíaca")
    st.markdown(
        "Demo del modelo entrenado en el proyecto **Hearth-project**. "
        "Ingrese los resultados del examen y el modelo estimará la probabilidad de "
        "enfermedad coronaria."
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
        st.dataframe(resultados, use_container_width=True)
        st.caption(
            "El modelo se seleccionó comparando 5 familias con validación cruzada y una "
            "prueba estadística (t-test corregido de Nadeau-Bengio)."
        )

    datos_paciente = construir_formulario()

    st.divider()
    if st.button("Predecir", type="primary", use_container_width=True):
        probabilidad = modelo.predict_proba(datos_paciente)[0, 1]
        mostrar_resultado(probabilidad)

        with st.expander("Ver los datos enviados al modelo"):
            st.dataframe(datos_paciente, use_container_width=True)


if __name__ == "__main__":
    main()
