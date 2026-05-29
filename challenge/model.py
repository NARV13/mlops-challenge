# Importar librerías para el modelo de ML
import os
import sys
import pickle
import pandas as pd

# CONFIGURACIÓN CRÍTICA PARA EVITAR BLOQUEOS EN POWERSHELL (PS)
os.environ["PYTHONIOENCODING"] = "utf-8"

# SonarQube S1192: Constante global para evitar duplicar el string literal en el código
COLUMNS_FILE_NAME = "columns.pkl"


class AnxietyPredictionModel:
    def __init__(self):
        self.model_name = "Modelo_de_prediccion_de_la_ansiedad"
        self.experiment_name = "Ansiedad_Adolescente_Tracking"
        self.model = None
        self.features_columns = None

    def load_data(self, file_path: str = "data_local.csv") -> pd.DataFrame:
        """Carga los datos de entrenamiento desde un archivo local seguro."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"🚨 No se encontró el archivo local '{file_path}'. "
                "Asegúrate de exportar tus datos a CSV antes de entrenar."
            )
        return pd.read_csv(file_path)

    def train_and_evaluate(self, df: pd.DataFrame):
        """Entrena, evalúa y registra ambos modelos en el tracker local de MLflow."""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        import mlflow.sklearn

        # Buscar la columna objetivo de ansiedad
        columna_target = None
        for col in df.columns:
            if "anxiety" in col.lower():
                columna_target = col
                break
        
        if not columna_target:
            raise ValueError("No se encontró la columna de nivel de ansiedad en el dataset.")

        # Preprocesamiento seguro por One-Hot Encoding
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop(columns=[c for c in df_encoded.columns if columna_target in c])
        y = df[columna_target]
        
        # Guardar el listado de columnas usando la constante global para producción
        self.features_columns = X.columns.tolist()
        with open(COLUMNS_FILE_NAME, "wb") as f:
            pickle.dump(self.features_columns, f)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Registro del modelo en MLflow
        with mlflow.start_run(run_name="Random_Forest_Modelo"):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_leaf=2, 
        max_features='sqrt')
            rf_model.fit(X_train, y_train)
            preds_rf = rf_model.predict(X_test)
            acc_rf = accuracy_score(y_test, preds_rf)
            
            mlflow.log_metric("exactitud", acc_rf)
            mlflow.sklearn.log_model(
                sk_model=rf_model, 
                artifact_path="modelo", 
                registered_model_name=self.model_name
            )
            print(f"   -> [MLflow] Random Forest registrado (Accuracy: {acc_rf:.4f})")

    def load_model_for_inference(self):
        """Carga dinámicamente el modelo activo desde el Registry de MLflow y recupera las columnas estructuradas."""
        import mlflow.sklearn
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
            # Cargar el modelo desde el Model Registry
            try:
                model_uri = f"models:/{self.model_name}@production"
                self.model = mlflow.sklearn.load_model(model_uri)
            except Exception:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                latest_versions = client.get_latest_versions(self.model_name)
                if latest_versions:
                    self.model = mlflow.sklearn.load_model(latest_versions[0].source)
                else:
                    raise LookupError("No se encontraron versiones registradas para este modelo en MLflow.")
            
            # Búsqueda robusta y absoluta del archivo usando la constante global
            base_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(base_dir, COLUMNS_FILE_NAME),
                os.path.join(base_dir, "..", COLUMNS_FILE_NAME),
                COLUMNS_FILE_NAME
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        self.features_columns = pickle.load(f)
                    print(f"✅ Éxito: Columnas de entrenamiento recuperadas desde {path}.")
                    break
                    
            if not self.features_columns:
                print(f"⚠️ Advertencia: No se pudo localizar el archivo '{COLUMNS_FILE_NAME}' en las rutas buscadas.")
                
            print("✅ Modelo cargado correctamente en memoria RAM.")
        except Exception as e:
            raise RuntimeError(f"Falla crítica en la inicialización de la inferencia: {e}")

    def predict(self, data: pd.DataFrame) -> list:
        """Realiza predicciones aplicando el reindexado con las columnas de entrenamiento."""
        if self.model is None:
            raise ValueError("El modelo de producción no está cargado en memoria.")
        
        # Generar las columnas del One-Hot encoding para la data entrante
        data_encoded = pd.get_dummies(data)
        
        # Alinear el DataFrame entrante con las columnas exactas del entrenamiento
        if self.features_columns:
            data_encoded = data_encoded.reindex(columns=self.features_columns, fill_value=0)
            
        raw_predictions = self.model.predict(data_encoded)
        return [int(pred) for pred in raw_predictions]


if __name__ == "__main__":
    print("📢 ¡El archivo model.py se está ejecutando en PowerShell exitosamente!")
    pipeline = AnxietyPredictionModel()
    try:
        pipeline.load_model_for_inference()
        if pipeline.features_columns:
            sample_df = pd.DataFrame([0] * len(pipeline.features_columns)).T
            sample_df.columns = pipeline.features_columns
            res = pipeline.predict(sample_df)
            print(f"🎉 ¡Prueba exitosa! Predicción: {res}")
    except Exception as e:
        print(f"ℹ️ Nota de inicialización: {e}")