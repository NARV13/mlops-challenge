import os
import sys
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
        extra = "ignore"  # Ignora campos dinámicos residuales de Swagger como additionalProp1


class PredictionRequest(BaseModel):
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Lista de niveles de ansiedad predichos por el modelo.")


# EVENTOS DE CICLO DE VIDA (Lifecycle Events)

@app.on_event("startup")
def startup_event():
    try:
        # Carga nativa del modelo y recuperación del listado columns.pkl
        predictor.load_model_for_inference()
    except Exception as e:
        print(f"🚨 Alerta en startup: No se pudo precargar desde MLflow. Detalle: {e}")


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
        # 1. Convertir datos validados de Pydantic a lista de diccionarios
        raw_data = [patient.dict() for patient in payload.patients]
        df_input = pd.DataFrame(raw_data)
        
        # 2. Convertir las columnas a minúsculas para que coincidan con el dataset original de entrenamiento
        # Ejemplo: 'Age' -> 'age', 'Gender' -> 'gender'
        df_input.columns = df_input.columns.str.lower()
        
        # 3. Validar que la variable 'gender' mantenga consistencia con los strings del set de datos
        if "gender" in df_input.columns:
            df_input["gender"] = df_input["gender"].astype(str)
            
        # 4. Enviar el DataFrame preprocesado al pipeline interno de model.py
        result_predictions = predictor.predict(df_input)
        
        return PredictionResponse(predictions=result_predictions)
        
    except Exception as e_inference:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error durante el procesamiento o inferencia de los datos: {str(e_inference)}"
        )