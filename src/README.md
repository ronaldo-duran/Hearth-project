# Feature/Training/Inference Pipelines

## Pipelines FTI del proyecto

Los tres scripts son autónomos y se ejecutan desde la raíz del repositorio, en este orden:

| Paso | Comando | Entrada | Salida |
| --- | --- | --- | --- |
| 1. Features | `uv run python src/pipelines/feature_pipeline/feature_pipeline.py` | `data/01_raw/corazon.csv` | `data/03_primary/corazon_limpio.parquet` |
| 2. Entrenamiento | `uv run python src/pipelines/training_pipeline/train_pipeline.py` | features del paso 1 | `models/modelo_final.joblib` y `models/modelo_final_metadatos.joblib` |
| 3. Inferencia | `uv run python src/pipelines/inference_pipeline/inference_pipeline.py <entrada.csv> [salida.csv]` | CSV con pacientes nuevos | CSV con `probabilidad_enfermedad` y `prediccion` (por defecto `data/07_model_output/predicciones.csv`) |

Ejemplo de inferencia con el archivo incluido en el repositorio:

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py src/inference/ejemplos/pacientes_ejemplo.csv
```

Validaciones que detienen el pipeline antes de persistir un resultado inválido:

| Pipeline | Módulo | Qué valida |
| --- | --- | --- |
| Features | `feature_pipeline/data_validation.py` | Esquema, tipos, rangos, categorías, % de nulos y unicidad |
| Entrenamiento | `training_pipeline/split_validation.py` | Fuga de datos entre train y test y drift de distribuciones (Evidently) |
| Entrenamiento | `training_pipeline/model_validation.py` | Validación cruzada estratificada y diagnóstico de over/underfitting |
| Inferencia | `inference_pipeline/inference_pipeline.py` | Columnas requeridas y registros no vacíos |

Las rutas, los dominios válidos de los datos y el umbral de decisión están en `pipelines/config.py`.
Las pruebas unitarias se ejecutan con `uv run pytest --cov` y corren en el CI de cada Pull Request.

## Estructura de referencia

File Structure based on:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

## Folder Structure

- src: source code
    - data: data extraction, data validation, data processing, data transformation, data save and export, etc.
    - model: model training, model evaluation, model validation, model save and export, etc.
    - inference: model prediction, model serving, model monitoring, etc.
    - pipelines:
        - feature_pipeline: takes as input raw data that it transforms into features (and labels)
        - training_pipeline: takes as input features and labels that it transforms into a model
        - inference_pipeline: takes new feature data and a trained model and makes predictions.

you could have multiple pipelines, for example:

- 3 feature pipelines that extract raw data from different sources and transform them into features and save it into a feature store.
- 2 training pipelines that take the features from the feature store and train different models.
- 3 inference pipeline that creates a model serving endpoint for each of the trained models and 1 batch
  inference pipeline that takes the features from the feature store and makes predictions in batch mode.

Finally is recommended to have a script that orchestrates the execution of the pipelines. This script should could be run in a cron job or a workflow orchestrator like Airflow, Prefect, Dagster, etc.
