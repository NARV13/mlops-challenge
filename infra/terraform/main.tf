
# 1. CUENTA DE SERVICIO E IAM (Seguridad con Privilegios Mínimos)


# Crea la cuenta de servicio específica para ejecutar la API de Ansiedad
resource "google_service_account" "api_execution_sa" {
  account_id   = "anxiety-api-sa-${var.environment}"
  display_name = "Service Account para la API de Ansiedad - Ambiente ${var.environment}"
  project      = var.project_id
}

# Asigna permisos para que la cuenta de servicio pueda escribir logs en Google Cloud Logging
resource "google_project_iam_member" "logging_permissions" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.api_execution_sa.email}"
}



# 2. GOOGLE ARTIFACT REGISTRY (Almacenamiento de Imágenes Docker)


# Crea el repositorio privado para las imágenes de Docker de la aplicación
resource "google_artifact_registry_repository" "api_repository" {
  project       = var.project_id
  location      = var.region
  repository_id = "anxiety-api-repo-${var.environment}"
  description   = "Repositorio Docker para la API de prediccion de ansiedad - ${var.environment}"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    project     = "mlops-challenge"
  }
}



# 3. GOOGLE CLOUD RUN (Servicio de Cómputo Serverless)


# Despliega el contenedor de la API en Cloud Run
resource "google_cloud_run_v2_service" "api_service" {
  name     = "anxiety-api-service-${var.environment}"
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL" # Permite solicitudes HTTP directas desde internet

  template {
    # Vincula la cuenta de servicio de ejecución creada arriba
    service_account = google_service_account.api_execution_sa.email

    containers {
      # Apuesta temporalmente a la imagen del repositorio (se sobreescribe en el pipeline de CI/CD)
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.api_repository.repository_id}/anxiety-api:latest"

      ports {
        container_port = 8080 # Puerto interno en el que escucha FastAPI
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi" # Memoria recomendada para almacenar el modelo MLflow en RAM
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
    }

    scaling {
      min_instance_count = 0 # Escala a 0 cuando no hay tráfico para reducir costos a cero
      max_instance_count = 3 # Límite de instancias consecutivas en picos de carga
    }
  }

  # Hace que Terraform ignore actualizaciones manuales del tag de la imagen hechas por el CI/CD
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }
}



# 4. CONFIGURACIÓN DE RED E INGRESO PÚBLICO (Ingress Público)


# Habilita el acceso HTTP público (sin tokens de GCP) para revisión de dev, qa o prod
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api_service.name
  role     = "roles/run.invoker"
  member   = "allUsers" # Define acceso abierto global bajo HTTPS cifrado
}