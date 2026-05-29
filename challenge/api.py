import os
import sys
import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from challenge.model import AnxietyPredictionModel

# Forzar codificación limpia para la consola de Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

app = FastAPI(
    title="API de Predicción de Ansiedad Adolescente",
    description="Servicio en producción para evaluar y predecir niveles de ansiedad basados en variables clínicas y demográficas.",
    version="1.0.0"
)

# Instanciar el motor del modelo global
predictor = AnxietyPredictionModel()

# SonarQube S1192: Constante global para evitar duplicar el string literal en el código
COLUMNS_ARTIFACT_NAME = "columns.pkl"


# MODELOS DE PYDANTIC (Esquema de entrada exacto para el Swagger)

class PatientData(BaseModel):
    """
    Formatos originales de variables requeridas por la interfaz de usuario.
    Alineado con el JSON predeterminado de Swagger.
    """
    Age: int = Field(..., description="Edad del paciente", example=16)
    Gender: str = Field(..., description="Género del paciente ('M' o 'F')", example="M")
    Sleep_Duration: float = Field(..., description="Horas de sueño promedio diarias", example=6.5)
    School_Pressure: int = Field(..., description="Nivel de presión escolar (1-10)", example=7)
    Study_Satisfaction: int = Field(..., description="Nivel de satisfacción con el estudio (1-10)", example=5)

    class Config:
        extra = "ignore"  # Ignora campos dinámicos residuales de Swagger


class PredictionRequest(BaseModel):
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Lista de niveles de ansiedad predichos por el modelo.")


# FUNCIONES DE SOPORTE (Reduce Complejidad Cognitiva - SonarQube S3776)

def find_local_artifact_path(base_directory: str, filename: str) -> str:
    """Busca de forma segura un archivo en las distintas rutas posibles del contenedor."""
    parent_directory = os.path.dirname(base_directory)
    possible_paths = [
        os.path.join(parent_directory, filename),
        os.path.join(base_directory, filename),
        filename
    ]
    for current_path in possible_paths:
        if os.path.exists(current_path):
            return current_path
    return ""


# EVENTOS DE CICLO DE VIDA (Lifecycle Events)

@app.on_event("startup")
def startup_event():
    try:
        print("⏳ API Iniciando: Intentando conectar al Model Registry de MLflow...")
        predictor.load_model_for_inference()
    except Exception as e:
        print(f"🚨 Alerta en startup: No se pudo conectar a MLflow de forma directa. Detalle: {e}")
        print("📦 Iniciando plan de contingencia: Cargando artefactos locales...")
        
        try:
            import mlflow.sklearn
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 1. Localizar y cargar el modelo físico localmente
            local_model_path = find_local_artifact_path(base_dir, "best_model")
            if local_model_path:
                predictor.model = mlflow.sklearn.load_model(local_model_path)
                print(f"🚀 ¡Éxito de contingencia! Modelo cargado desde: {local_model_path}")
            else:
                print("❌ Error crítico de contingencia: No se encontró la carpeta 'best_model' en ninguna ruta.")

            # 2. Localizar y cargar las columnas físicas locales (EVITA EL ERROR 422)
            local_columns_path = find_local_artifact_path(base_dir, COLUMNS_ARTIFACT_NAME)
            if local_columns_path:
                with open(local_columns_path, "rb") as f:
                    predictor.features_columns = pickle.load(f)
                print(f"✅ ¡Éxito de contingencia! Columnas fijadas desde {local_columns_path} ({len(predictor.features_columns)} variables).")
            else:
                print(f"❌ Error crítico de contingencia: No se encontró '{COLUMNS_ARTIFACT_NAME}' en ninguna ubicación.")

        except Exception as e_local:
            print(f"🚨 Falla crítica absoluta: El plan de contingencia local también falló: {e_local}")


# ENDPOINTS HTTP

@app.get("/health")
def health_check():
    status_modelo = "Listo y Conectado" if predictor.model is not None else "No Inicializado"
    return {
        "status": "Healthy",
        "model_status": status_modelo,
        "environment": "Production"
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inferencia"])
def predict_anxiety(payload: PredictionRequest):
    if predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de Machine Learning no está inicializado en el servidor."
        )
    
    try:
        # 1. Convertir datos validados de Pydantic a lista de diccionarios planos (evita errores unhashable)
        raw_data = [patient.dict() for patient in payload.patients]
        df_input = pd.DataFrame(raw_data)
        
        # 2. Transformar las columnas entrantes a minúsculas para alinearlas con el dataset de entrenamiento
        df_input.columns = df_input.columns.str.lower()
        
        # 3. Forzar tipo string seguro para la variable categórica género
        if "gender" in df_input.columns:
            df_input["gender"] = df_input["gender"].astype(str)
            
        # 4. Enviar el DataFrame preprocesado al pipeline interno del modelo
        result_predictions = predictor.predict(df_input)
        
        return PredictionResponse(predictions=result_predictions)
        
    except Exception as e_inference:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error durante el procesamiento o inferencia de los datos: {str(e_inference)}"
        )