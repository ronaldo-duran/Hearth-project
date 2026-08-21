# Demo funcional - Predicción de enfermedad cardíaca

Formulario web que expone el modelo entrenado en el proyecto. El usuario ingresa los
resultados de un examen médico y el modelo devuelve la probabilidad de enfermedad
coronaria.

Corresponde al [issue #15](https://github.com/ronaldo-duran/Hearth-project/issues/15).

## Cómo ejecutarlo

Desde la raíz del repositorio:

```bash
uv sync
```

```bash
uv run streamlit run src/inference/app.py
```

La aplicación queda disponible en <http://localhost:8501> y se abre sola en el navegador.

Para ejecutarla sin que abra el navegador (por ejemplo en un servidor):

```bash
uv run streamlit run src/inference/app.py --server.headless true
```

## Qué necesita para funcionar

La app carga dos artefactos que produce el notebook del
[issue #13](https://github.com/ronaldo-duran/Hearth-project/issues/13), ambos versionados
en el repositorio:

| Archivo | Contenido |
| --- | --- |
| `models/modelo_final.joblib` | Pipeline completo: preprocesamiento + Random Forest |
| `models/modelo_final_metadatos.joblib` | Algoritmo, métrica principal y resultados en test |

Como el `.joblib` contiene el **pipeline completo**, el formulario envía los datos en su
formato **crudo** (`Male`, `asymptomatic`, `reversable`…) y el propio pipeline se encarga
de imputar, escalar y codificar. No hay ninguna transformación replicada a mano en la app,
que es justamente lo que evita que la demo y el entrenamiento se desincronicen.

## Publicar en Streamlit Community Cloud

El repositorio ya cumple los requisitos: es público, los `.joblib` están versionados y
`requirements.txt` fija las versiones exactas con las que se serializó el modelo.

1. Entrar a <https://share.streamlit.io> e iniciar sesión con la cuenta de GitHub.
2. **New app** → **Deploy a public app from GitHub**.
3. Completar:
   - Repository: `ronaldo-duran/Hearth-project`
   - Branch: `main`
   - Main file path: `src/inference/app.py`
4. En **Advanced settings**, elegir Python **3.12** (el del proyecto, según `.python-version`).
5. **Deploy**. La primera instalación tarda unos minutos.

> **Importante**: no fue posible usar `pyproject.toml` porque Streamlit Community Cloud no
> instala dependencias con uv. Por eso existe `requirements.txt` en la raíz, con las
> versiones **fijadas exactamente**: un `scikit-learn` distinto al que serializó
> `modelo_final.joblib` puede fallar al deserializarlo o cambiar su comportamiento sin
> avisar.

## Cómo se usa

1. Complete los 13 campos del examen (todos tienen un valor por defecto razonable).
2. Pulse **Predecir**.
3. La app muestra la probabilidad estimada, la decisión según el umbral y, en un
   desplegable, los datos exactos que se enviaron al modelo.

## El umbral de decisión

La app usa un umbral de **0.411** en lugar del 0.5 por defecto. No es arbitrario: sale del
análisis del [issue #14](https://github.com/ronaldo-duran/Hearth-project/issues/14).

En screening cardíaco el falso negativo (dar por sano a un paciente enfermo) es mucho más
grave que el falso positivo (mandar a un sano a un examen adicional). Bajar el umbral sube
el recall de 0.891 a 0.913 a costa de precision, y ese intercambio es el correcto para este
problema.

El umbral está en la constante `UMBRAL_DECISION` de `app.py`, para que se pueda ajustar si
el criterio clínico cambia.

## Desempeño del modelo

| Métrica | Test |
| --- | --- |
| accuracy | 0.854 |
| recall | 0.891 |
| ROC-AUC | 0.928 |

## Aviso

Es una **herramienta académica**, no un dispositivo médico. El modelo se entrenó con 480
pacientes del estudio Cleveland (UCI) de los años 80 y no está validado clínicamente. No
sustituye el criterio de un profesional de la salud.
