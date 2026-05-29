### Use of AI tools

### Exploración de datos
se hizo uso de la AI GEMINI para el apartado de exploración de datos  en cuanto a la sintaxis para la visualización de la base de datos que se encontraba en bigquery, en el apartado de modelos se uso para la creación del modelo de random forest. 

En el apartado de EDA se realiza, verificación de tamaño de los datos, verificación de datos nulos, resumen de las columnas, tipos de datos y si hay valores nulos, estadisticas principales como media, mediana , desviación estándar, minimo, máximo, revisión de variables tipo objeto o categorica, histogramas, grafico de barras, mapa de calor, cambio de bbariabbles tipo texto a variables númericas, definición de variable objeto, eliminación de algunas columnas categóricas y nos centraremos en las características numéricas, modelo de regresión logística.

comparación de modelos (Random forest y Logistic Regression)

### Tracker Seleccionado (Realizado con AI GEMINI)
Se seleccionó **MLflow** como la plataforma central para el seguimiento de experimentos, gestión de métricas y registro de artefactos. Su elección se debe a su ligereza para la infraestructura local en VS Code y su compatibilidad nativa con el ciclo de vida de MLOps, permitiendo una transición transparente hacia servidores gestionados (como Vertex AI Experiments o Databricks) en fases posteriores. (ayuda de la AI)

### Análisis de Experimentos y Selección de Modelo (Realizado con AI GEMINI)

Durante la fase de experimentación y automatización, se evaluaron dos arquitecturas distintas cuyos resultados históricos quedaron registrados en el servidor de MLflow:

| Algoritmo | Hiperparámetros Evaluados | Accuracy Obtenido | Estado | Juicio Técnico |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `max_iter=1000`, `random_state=42` | **79.6%** | **Seleccionado para Producción** | Excelente capacidad de generalización. La naturaleza del nivel de ansiedad (escala ordinal/progresiva) posee un comportamiento lineal que el modelo captura con precisión matemática. |
| **Random Forest Classifier** | `n_estimators=100`, `random_state=42` | **28.3%** | **Descartado** | Rendimiento deficiente (cercano al azar para 10 clases). Al ser un modelo basado en árboles, trata los niveles como categorías aisladas e inconexas, perdiendo la noción de escala o jerarquía numérica. |
---

### Cómo Inspeccionar los Experimentos / Runs (Realizado con AI GEMINI)

Para auditar, comparar o visualizar las métricas e hiperparámetros guardados localmente, siga estos pasos:

1. Asegúrese de activar el entorno virtual y ejecute el pipeline de entrenamiento para generar los registros actualizados
2. ingrese a mlfllow en el apartado experimentos y abra "Ansiedad_Adolescente_Tracking"
3. 

### Registrar Modelos (Realizado con AI GEMINI)

según la guía se asigna al modelo random forest el alias de staging y al modelo de regresión logistica al de production. (ayuda de la AI)

### Modelo.py (Realizado con AI GEMINI)

Esta sección fue con ayuda de la AI ya que no me queria correr el código por lo cual la AI me dio arrojo un código con mayor presión y funcionamiento con la terminal de VS Code.

---

## Módulo API: FastAPI & OpenAPI Specification (Realizado con AI GEMINI)

### 1. Disponibilidad de Interfaces (Realizado con AI GEMINI)
La solución expone los siguientes endpoints automáticos interactivos para pruebas de integración y autoservicio de desarrollo:
* **Swagger UI (Interactive Docs):** Disponible localmente en `http://127.0.0.1:8000/docs`. Permite realizar solicitudes `POST /predict` directamente desde el navegador de manera visual.
* **ReDoc (Alternative Docs):** Disponible en `http://127.0.0.1:8000/redoc`, enfocado en una lectura limpia de la documentación estructurada.
* **Esquema Crudo OpenAPI:** Generado dinámicamente en formato JSON en `http://127.0.0.1:8000/openapi.json`.

### 2. Descarga Automatizada del Esquema (Contrato de API) (Realizado con AI GEMINI)
Para exportar el esquema OpenAPI estándar de la aplicación (útil para pipelines de API Gateway o generación de SDKs clientes), se puede utilizar cualquiera de las siguientes herramientas de terminal sin necesidad de interactuar con la interfaz gráfica:

## Despliegue con Docker y Contenedores (Realizado con AI GEMINI)

La aplicación ha sido completamente contenedorizada para garantizar la reproducibilidad y el aislamiento ambiental de la arquitectura MLOps.

### 1. Construcción de la Imagen (Docker Build) (Realizado con AI GEMINI)
Para construir la imagen Docker local de la API, ejecuta el siguiente comando desde la raíz del proyecto (`mlops-challenge/`):

# Construir la imagen local de Docker (Realizado con AI GEMINI)
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" build --no-cache -t anxiety-api:1.0.0 .

### Desplegar el contenedor mapeando el puerto de escucha externa de la API (Realizado con AI GEMINI)
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" run -p 8080:8080 --name anxiety-api-service anxiety-api:1.0.0

--no-cache: Fuerza a Docker a reconstruir cada capa desde cero, garantizando la lectura de los últimos archivos físicos de best_model/.

-t anxiety-api:1.0.0: Etiqueta la imagen con un nombre claro y una versión semántica específica.

### Cómo Ejecutar el Contenedor Localmente (Realizado con AI GEMINI)

# 1. Eliminar contenedores previos con el mismo nombre (evita conflictos) (Realizado con AI GEMINI)
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" rm -f anxiety-api-service

# 2. Lanzar el contenedor exponiendo el puerto 8080 (Realizado con AI GEMINI)
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" run -p 8080:8080 --name anxiety-api-service anxiety-api:1.0.0

Nota: La terminal se quedará bloqueada mostrando los logs de ejecución de Uvicorn en tiempo real. Para validar un inicio exitoso, busca la siguiente línea en los logs:

INFO: Application startup complete.

### Matriz de Verificación de Endpoints (Realizado con AI GEMINI)
Ruta de Salud (/health)http://localhost:8080/healthDevuelve un JSON confirmando el estado de la API y que el modelo está cargado en memoria RAM.
Documentación Interactiva (/docs)http://localhost:8080/docsInterfaz gráfica de Swagger UI. Permite realizar pruebas en tiempo real de los endpoints (POST /predict) directamente desde el navegador.
Contrato Técnico (/openapi.json)http://localhost:8080/openapi.jsonGenera el esquema matemático completo en formato OpenAPI, detallando los tipos de datos de entrada y salida requeridos por el modelo.

## Infraestructura y Entornos (GCP & Terraform) (Realizado con AI GEMINI)

Para garantizar el aislamiento total de recursos, la seguridad y un ciclo de vida de desarrollo robusto, la infraestructura se administra como código (IaC) utilizando **Terraform** y se despliega en proyectos independientes de Google Cloud Platform (GCP).

Toda la infraestructura está configurada para desplegarse en la región **`southamerica-east1` (Bogotá)** con el fin de optimizar costos y latencia.

### Matriz de Configuración de Proyectos (Realizado con AI GEMINI)

| Entorno | ID del Proyecto en GCP (Project ID) | Región Principal | Propósito / Rama Git | URL Pública de la API |
| :--- | :--- | :--- | :--- | :--- |
| **Development** | `anxiety-prediction-dev` | `southamerica-east1` | Experimentación (`feature/*`) | `https://anxiety-api-dev-xxxx-rj.a.run.app` |
| **QA / Testing** | `anxiety-prediction-qa` | `southamerica-east1` | Pruebas integrales (`develop`) | `https://anxiety-api-qa-xxxx-rj.a.run.app` |
| **Production** | `anxiety-prediction-prod` | `southamerica-east1` | Cliente Final (`main`) | `https://anxiety-api-prod-xxxx-rj.a.run.app` |


### Estructura del Directorio de Infraestructura (Realizado con AI GEMINI)

El código de Terraform se mantiene estrictamente separado de la lógica de la aplicación dentro del directorio `infra/`:

infra/
└── terraform/
    ├── main.tf          # Definición de recursos (Cloud Run, Artifact Registry, IAM)
    ├── variables.tf     # Declaración de variables (project_id, region, environment)
    ├── outputs.tf       # Retorno de datos críticos (URLs de Cloud Run, rutas de imágenes)
    ├── terraform.tfvars # Valores por defecto de las variables
    └── providers.tf     # Configuración del proveedor de Google Cloud

### Validación Local de la Infraestructura (Pruebas de Terraform) (Realizado con AI GEMINI)

Para garantizar que el código de Terraform es correcto y está listo para ser procesado por un pipeline de CI/CD sin romper la infraestructura, se puede simular el despliegue de manera 100% local sin necesidad de contar con permisos reales de escritura en Google Cloud en el entorno de desarrollo.

### Pasos para Validar el Código (Realizado con AI GEMINI)

1. **Inicializar el Entorno:**
Descarga los plugins y conectores oficiales de Google Cloud (`hashicorp/google`) requeridos por la arquitectura.
cd infra/terraform
./terraform init

2. Validación Sintáctica: (Realizado con AI GEMINI)
Analiza el archivo main.tf buscando errores de tipeo, llaves sin cerrar o variables no declaradas.

./terraform validate

3. Simulación del Plan de Acción (terraform plan): (Realizado con AI GEMINI)
Utilizando un archivo de credenciales ficticio (fake-credentials.json) y la bandera -refresh=false, se le ordena a Terraform simular qué recursos crearía en el entorno de desarrollo (dev) sin tocar internet:

./terraform plan -var="project_id=anxiety-prediction-dev" -var="environment=dev" -refresh=false

4. Resultado Exitoso del Plan (Matriz de Recursos) (Realizado con AI GEMINI)
Una ejecución correcta de la simulación debe arrojar el mensaje Plan: 5 to add, 0 to change, 0 to destroy.

## Ciclo de CI/CD y Flujo de Código (GitOps) (Realizado con AI GEMINI)

El proyecto implementa una arquitectura de Integración Continua (CI) y Despliegue Continuo (CD) completamente automatizada con **GitHub Actions**, garantizando que cada cambio en el código sea validado estricta y analíticamente antes de tocar cualquier entorno de Google Cloud.

### Flujo de Código a través de Entornos (Realizado con AI GEMINI)


El código transita de manera lineal y segura a través de las ramas del repositorio, disparando despliegues automáticos e inmutables:

 [Rama feature/*] ───► [PR a develop] ───► [Rama develop] ───► [Rama release/*] ───► [Rama main / Tag v*]
        │                     │                   │                     │                     │
   Local / Dev            CI Validation       CD to DEV             CD to QA              CD to PROD
 (Pruebas Locales)     (Tests & Sonar)     (GCP Dev Project)     (GCP QA Project)      (GCP Prod Project)

