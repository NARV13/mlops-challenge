#Import librerias para el modelo de ML
import os
import sys
import pickle
import pandas as pd
import google.oauth2.credentials
from google.cloud import bigquery

# CONFIGURACIÓN CRÍTICA PARA EVITAR BLOQUEOS EN POWERSHELL (PS)
os.environ["PYTHONIOENCODING"] = "utf-8"
# Deshabilitar instrumentaciones asíncronas que congelan hilos en Windows
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
os.environ["GOOGLE_CLOUD_DISABLE_GRPC"] = "True"

print("📢 ¡El archivo train.py se está ejecutando en PowerShell exitosamente!")
print("1/4. Cargando módulos de conexión a BigQuery...")
def main():
    # Token temporal para Google Cloud BigQuery
    TOKEN_TEMPORAL = os.getenv("GOOGLE_BIGQUERY_TOKEN", "TU_TOKEN_TEMPORAL_AQUI")
    print("2/4. Conectando y descargando datos de BigQuery...")
    credenciales = google.oauth2.credentials.Credentials(TOKEN_TEMPORAL)
    client = bigquery.Client(project='teen-mental-health-497520', credentials=credenciales)
    query = "SELECT * FROM `teen-mental-health-497520.ML_dataset.Pacientes`"
    
    try:
        query_job = client.query(query)
        resultados = query_job.result()  
        filas = [dict(row) for row in resultados]
        df = pd.DataFrame(filas)
        print(f"   -> ¡Datos descargados con éxito! Total de filas: {len(df)}")
        
        # Guardar una copia local en CSV para respaldar la estrategia de model.py
        df.to_csv("data_local.csv", index=False)
        print("   -> Respaldado local 'data_local.csv' generado correctamente.")
    except Exception as e:
        print(f"🚨 Error al descargar datos de BigQuery: {e}")
        return

    print("3/4. Cargando librerías de Machine Learning y MLflow...")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import mlflow
    import mlflow.sklearn

    # Forzar la ruta del servidor local antes de interactuar con el experimento
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Ansiedad_Adolescente_Tracking")

    # Identificar la columna target de ansiedad de forma dinámica
    columna_encontrada = None
    for col in df.columns:
        if "anxiety" in col.lower():
            columna_encontrada = col
            break
    
    if columna_encontrada is None:
        print(f"🚨 ERROR: No se encontró la columna objetivo. Columnas: {df.columns.tolist()}")
        return
        
    print(f"🎯 Variable objetivo identificada: '{columna_encontrada}'")
    
    # Preprocesamiento estructural
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=[c for c in df_encoded.columns if columna_encontrada in c])
    y = df[columna_encontrada]

    # Guardar orden de columnas estructurales
    with open("columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    # División de conjuntos (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("4/4. Ejecutando y registrando ejecuciones en el servidor local...")

    try:
        
        # EJECUCIÓN 1: REGRESIÓN LOGÍSTICA
        
        with mlflow.start_run(run_name="Regresion_Logistica_Modelo"):
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            mlflow.log_param("tipo_de_modelo", "Regresión Logística")
            
            log_reg.fit(X_train, y_train)
            y_pred_lr = log_reg.predict(X_test)
            acc_lr = accuracy_score(y_test, y_pred_lr)
            
            mlflow.log_metric("exactitud", acc_lr)
            print(f"   -> [MLflow] Regresión Logística entrenada con éxito (Acc: {acc_lr:.4f})")
            
            mlflow.sklearn.log_model(
                sk_model=log_reg, 
                artifact_path="modelo",
                registered_model_name="Modelo_de_prediccion_de_la_ansiedad"
            )

        
        # EJECUCIÓN 2: RANDOM FOREST
        
        with mlflow.start_run(run_name="Random_Forest_Modelo"):
            rand_forest = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_leaf=2, 
        max_features='sqrt')
            mlflow.log_param("tipo_de_modelo", "Random Forest")
            
            rand_forest.fit(X_train, y_train)
            y_pred_rf = rand_forest.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            
            mlflow.log_metric("exactitud", acc_rf)
            print(f"   -> [MLflow] Random Forest entrenado con éxito (Acc: {acc_rf:.4f})")
            
            mlflow.sklearn.log_model(
                sk_model=rand_forest, 
                artifact_path="modelo",
                registered_model_name="Modelo_de_prediccion_de_la_ansiedad"
            )

        print("\n==========================================================")
        print("🎉 ¡AMBOS MODELOS REGISTRADOS DESDE POWERSHELL CON ÉXITO!")
        print("==========================================================")

    except Exception as e_mlflow:
        print(f"\n🚨 Error de comunicación con el servidor MLflow: {e_mlflow}")
        print("👉 Asegúrate de tener el comando 'mlflow ui' corriendo en tu otra pestaña de PS.")

if __name__ == "__main__":
    main()