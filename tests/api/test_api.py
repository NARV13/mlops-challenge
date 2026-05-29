import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Asegurar que el entorno de testing maneje rutas limpias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importar la aplicación directamente desde challenge.api
from challenge.api import app

# Forzar la inicialización limpia del cliente utilizando un bloque de contexto seguro
client = TestClient(app, raise_server_exceptions=False)

# =========================================================================
# 1. PRUEBA DEL ENDPOINT /health
# =========================================================================
def test_health_endpoint():
    """Valida que el health check devuelva un código 200 y la estructura correcta."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "Healthy"
    assert "model_status" in json_data


# =========================================================================
# 2. PRUEBAS DEL ENDPOINT /predict
# =========================================================================
@patch("challenge.api.predictor")
def test_predict_endpoint_valid_payload(mock_predictor):
    """Valida que /predict procese un JSON correcto y devuelva las predicciones mockeadas."""
    # Configurar el comportamiento simulado de tu motor de predicción
    mock_predictor.model = MagicMock()  
    mock_predictor.predict.return_value = [1]  

    # Payload que cumple estrictamente con el esquema Pydantic (PatientData)
    valid_payload = {
        "patients": [
            {
                "Age": 16,
                "Gender": "M",
                "Sleep_Duration": 6.5,
                "School_Pressure": 7,
                "Study_Satisfaction": 5
            }
        ]
    }

    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)
    assert json_data["predictions"] == [1]


def test_predict_endpoint_invalid_payload():
    """Valida que /predict devuelva un código 422 (Unprocessable Entity) si faltan campos."""
    # Enviamos un payload vacío que viola las reglas de Pydantic
    invalid_payload = {"patients": [{}]}
    
    response = client.post("/predict", json=invalid_payload)
    
    # FastAPI intercepta esto automáticamente gracias a Pydantic y responde 422
    assert response.status_code == 422
    json_data = response.json()
    assert "detail" in json_data


@patch("challenge.api.predictor")
def test_predict_endpoint_model_unavailable(mock_predictor):
    """Valida que la API responda con un error 503 si el modelo no ha sido inicializado."""
    # Simular que el modelo no se encuentra cargado en memoria
    mock_predictor.model = None
    
    valid_payload = {
        "patients": [
            {
                "Age": 16,
                "Gender": "M",
                "Sleep_Duration": 6.5,
                "School_Pressure": 7,
                "Study_Satisfaction": 5
            }
        ]
    }

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert "inicializado" in response.json()["detail"].lower() or "initialised" in response.json()["detail"].lower() or "machine learning" in response.json()["detail"].lower()