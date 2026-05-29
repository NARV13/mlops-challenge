# =========================================================================
# 1. IMAGEN BASE OPTIMIZADA
# =========================================================================
# Usamos una versión ligera de Python 3.11 basada en Debian Slim
FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc en el contenedor
ENV PYTHONDONTWRITEBYTECODE=1
# Forzar que los logs salgan directamente a la consola sin almacenamiento en búfer
ENV PYTHONUNBUFFERED=1
# Configurar la codificación de caracteres predeterminada
ENV PYTHONIOENCODING=utf-8

# =========================================================================
# 2. CONFIGURACIÓN DEL DIRECTORIO DE TRABAJO
# =========================================================================
WORKDIR /app

# Instalar herramientas esenciales del sistema para compilación ligera
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================================================================
# 3. INSTALACIÓN DE DEPENDENCIAS
# =========================================================================
# Copiamos primero el archivo de requerimientos
COPY requirements.txt .

# Actualizar pip e instalar dependencias (incluyendo pandas, mlflow, etc.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Asegurar la instalación de servidores y clientes HTTP auxiliares
RUN pip install --no-cache-dir uvicorn httpx

# =========================================================================
# 4. COPIADO DE CÓDIGO Y ARTEFACTOS LOCALES
# =========================================================================
# Copiar el paquete modular de la aplicación
COPY challenge/ ./challenge/

# Copiar archivos de soporte y datos necesarios para el pipeline
COPY data_local.csv .
COPY columns.pkl .

# =========================================================================
# 5. EXPOSICIÓN DE PUERTOS Y COMANDO DE ARRANQUE
# =========================================================================
# Informar a Docker el puerto en el que escuchará el contenedor
EXPOSE 8080

# Comando de arranque utilizando Uvicorn.
# Escucha en 0.0.0.0 para permitir conexiones desde fuera del contenedor.
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]