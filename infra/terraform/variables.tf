variable "project_id" {
  type        = string
  description = "El ID del proyecto de Google Cloud Platform (GCP)"
}

variable "region" {
  type        = string
  description = "La región geográfica de GCP donde se desplegarán los recursos"
  default     = "southamerica-east1"
}

variable "environment" {
  type        = string
  description = "El entorno de despliegue (dev, qa, prod)"
}