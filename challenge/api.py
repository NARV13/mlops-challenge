#Importar las librerías necesarias para la API y el modelo de predicción
import os
import sys
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from challenge.model import AnxietyPredictionModel

# Forzar codificación limpia para la consola de Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

# Inicializar la aplicación FastAPI con metadatos para OpenAPI
app = FastAPI(
    title="API de Predicción de Ansiedad Adolescente",
    description="Servicio en producción para evaluar y predecir niveles de ansiedad basados en variables clínicas y demográficas.",
    version="1.0.0"
)

# Instanciar el motor del modelo global
predictor = AnxietyPredictionModel()


# MODELOS DE PYDANTIC (Validación de Esquemas del Request y Response)

class PatientData(BaseModel):
    """
    Representa el formato de características requerido para un paciente.
    FastAPI usará este modelo para generar automáticamente el esquema OpenAPI.
    """
    # Define aquí un ejemplo de los campos que vienen de tu BigQuery/Dataset.
    # Pydantic validará que los tipos de datos concuerden antes de tocar el modelo.
    Age: int = Field(..., description="Edad del paciente", example=16)
    Gender: str = Field(..., description="Género del paciente", example="M")
    Sleep_Duration: float = Field(..., description="Horas de sueño promedio diarias", example=6.5)
    School_Pressure: int = Field(..., description="Nivel de presión escolar percibida (1-10)", example=7)
    Study_Satisfaction: int = Field(..., description="Nivel de satisfacción con el estudio (1-10)", example=5)

    class Config:
        extra = "allow"  # Permite recibir columnas adicionales dinámicas del One-Hot Encoding


class PredictionRequest(BaseModel):
    """Contenedor para recibir uno o múltiples registros de pacientes en lote (Batch)."""
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    """Estructura estricta de la respuesta que retorna la API."""
    predictions: List[int] = Field(..., description="Lista de clases o niveles de ansiedad predichos por el modelo.")



# EVENTOS DE CICLO DE VIDA (Lifecycle Events)
@app.on_event("startup")
def startup_event():
    print("⏳ API Iniciando: Cargando artefacto matemático desde almacenamiento local...")
    try:
        # Importamos tu modelo directamente
        from challenge.model import AnxietyPredictionModel # Ajusta el nombre de tu instancia si es diferente
        
        # Forzamos la carga desde la carpeta física interna que Docker ya empaquetó
        import os
        import mlflow.pyfunc
        
        local_path = os.path.join(os.path.dirname(__file__), "best_model")
        if os.path.exists(local_path):
            AnxietyPredictionModel.model = mlflow.pyfunc.load_model(local_path)
            print("🚀 Éxito: ¡Modelo cargado en memoria RAM desde el empaquetado local!")
        else:
            print(f"🚨 Error: No se encontró la carpeta física del modelo en {local_path}")
            
    except Exception as e:
        print(f"🚨 Alerta crítica en startup: No se pudo precargar el modelo local. Error: {e}")
#@app.on_event("startup")
#def startup_event():
#    """Se ejecuta automáticamente al encender la API. Precarga el modelo desde MLflow @production."""
#    try:
#        print("⏳ API Iniciando: Cargando artefacto matemático desde el Model Registry...")
#        predictor.load_model_for_inference()
#        print("🚀 API Lista y modelo cargado exitosamente en memoria.")
#    except Exception as e:
#        print(f"🚨 Alerta crítica en startup: No se pudo precargar el modelo. {e}")
#        print("ℹ️ Nota: La API iniciará pero fallará en /predict hasta que MLflow esté disponible.")



# ENDPOINTS HTTP

@app.get("/health")
def health_check():
    """Verifica si la API está encendida y respondiendo."""
    return {
        "status": "Healthy",
        "model_status": "Cargado Exitosamente (Autocontenido)",
        "environment": "Production"
    }

#@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoreo"])
#def health_check():
#    """Verifica la disponibilidad básica del servicio (Health Check)."""
#    # Si el modelo no se cargó en el startup, reportamos un estado degradado
#    status_modelo = "Listo" if predictor.model is not None else "No Cargado (Verificar MLflow UI)"
#    return {
#        "status": "Healthy",
#        "model_status": status_modelo,
#        "environment": "Production"
#    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inferencia"])
def predict_anxiety(payload: PredictionRequest):
    """
    Recibe un lote de datos de pacientes, ejecuta el preprocesamiento estructural
    y devuelve las predicciones generadas por el modelo en producción.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de Machine Learning no está inicializado en el servidor."
        )
    
    try:
        # Convertir la lista de Pydantic directamente a un DataFrame de Pandas
        raw_data = [patient.dict() for patient in payload.patients]
        df_input = pd.DataFrame(raw_data)
        
        # Ejecutar la inferencia a través del pipeline de model.py
        result_predictions = predictor.predict(df_input)
        
        return PredictionResponse(predictions=result_predictions)
        
    except Exception as e_inference:
        raise HTTPException(
            status_code=status.HTTP_420_METHOD_FAILURE,
            detail=f"Error durante el procesamiento o inferencia de los datos: {str(e_inference)}"
        )