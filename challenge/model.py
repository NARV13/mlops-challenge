#Import librerias para el modelo de ML
import os
import sys
import pickle
import mlflow.pyfunc
import pandas as pd


# CONFIGURACIÓN CRÍTICA PARA EVITAR BLOQUEOS EN POWERSHELL (PS)

os.environ["PYTHONIOENCODING"] = "utf-8"

class AnxietyPredictionModel:
    def __init__(self):
        self.model_name = "Modelo_de_prediccion_de_la_ansiedad"
        self.experiment_name = "Ansiedad_Adolescente_Tracking"
        self.model = None
        self.features_columns = None

    def load_model_for_inference(self):
        print("⏳ API Iniciando: Cargando artefacto matemático...")
        
        # Definimos la ruta local interna del contenedor
        local_path = os.path.join(os.path.dirname(__file__), "best_model")
        
        if os.path.exists(local_path):
            try:
                print("📦 Cargando modelo desde artefactos locales autocontenidos...")
                # Cargamos el modelo localmente sin tocar la red
                self.model = mlflow.pyfunc.load_model(local_path)
                print("🚀 Modelo cargado exitosamente desde empaquetado local.")
                return
            except Exception as e:
                print(f"🚨 Error al cargar el modelo local: {e}")
        else:
            print(f"❌ No se encontró la carpeta local en: {local_path}")
            # Aquí puedes dejar tu código de MLflow viejo solo como un fallback por si acaso
    
    #def _init_mlflow(self):
    #    """Inicializa MLflow de forma perezosa para evitar conflictos de hilos."""
    #    import mlflow
    #    mlflow.set_experiment(self.experiment_name)
    #    return mlflow

    def load_data(self, file_path: str = "data_local.csv") -> pd.DataFrame:
        """Carga los datos de entrenamiento desde un archivo local seguro,
        evitando bloqueos de red en la terminal."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"🚨 No se encontró el archivo local '{file_path}'. "
                "Asegúrate de exportar tus datos a CSV antes de entrenar."
            )
        return pd.read_csv(file_path)

    def train_and_evaluate(self, df: pd.DataFrame):
        """Entrena, evalúa y registra ambos modelos en el tracker local de MLflow."""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        import mlflow.sklearn

        mlflow = self._init_mlflow()

        # Localizar dinámicamente la columna objetivo
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
        
        # Guardar el listado de columnas para producción
        self.features_columns = X.columns.tolist()
        with open("columns.pkl", "wb") as f:
            pickle.dump(self.features_columns, f)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # --- RUN 1: REGRESIÓN LOGÍSTICA ---
        with mlflow.start_run(run_name="Regresion_Logistica_Modelo"):
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            mlflow.log_param("tipo_de_modelo", "Regresión Logística")
            lr_model.fit(X_train, y_train)
            preds = lr_model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("exactitud", acc)
            
            mlflow.sklearn.log_model(
                sk_model=lr_model, 
                artifact_path="modelo", 
                registered_model_name=self.model_name
            )
            print(f"   -> [MLflow] Regresión Logística registrada (Accuracy: {acc:.4f})")

        # --- RUN 2: RANDOM FOREST ---
        with mlflow.start_run(run_name="Random_Forest_Modelo"):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            mlflow.log_param("tipo_de_modelo", "Random Forest")
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
        """Carga dinámicamente el modelo activo desde el Registry usando el alias @production."""
        import mlflow.sklearn
        try:
            # Guiar a MLflow hacia la dirección del servidor local
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
            model_uri = f"models:/{self.model_name}@production"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            if os.path.exists("columns.pkl"):
                with open("columns.pkl", "rb") as f:
                    self.features_columns = pickle.load(f)
            print("✅ Modelo en producción cargado correctamente desde MLflow Registry.")
        except Exception as e:
            raise RuntimeError(f"Error al conectar con MLflow Server para descargar el modelo: {e}")

    def predict(self, data: pd.DataFrame) -> list:
        """Realiza predicciones estructuradas para uno o múltiples registros de entrada.
        Devuelve una lista de enteros nativos válidos para respuestas JSON."""
        if self.model is None:
            raise ValueError("El modelo de producción no está cargado en memoria.")
        
        # Asegurar One-Hot Encoding idéntico al entrenamiento
        data_encoded = pd.get_dummies(data)
        
        # Reindexar para garantizar que tenga exactamente las mismas columnas y orden que vio el modelo
        if self.features_columns:
            data_encoded = data_encoded.reindex(columns=self.features_columns, fill_value=0)
            
        # Obtener las predicciones base de numpy
        raw_predictions = self.model.predict(data_encoded)
        
        # Convertir a tipos de datos nativos de Python (int) para evitar fallos de serialización JSON en la API
        return [int(pred) for pred in raw_predictions]



# VERIFICACIÓN LOCAL DESACOPLADA DE RED

if __name__ == "__main__":
    print("📢 ¡El archivo model.py se está ejecutando en PowerShell exitosamente!")
    
    pipeline = AnxietyPredictionModel()
    
    try:
        print("⏳ Conectando al Model Registry local...")
        pipeline.load_model_for_inference()
        
        print("⏳ Validando formato de la función predict()...")
        if pipeline.features_columns:
            # Crear un dataframe dummy basado en las columnas de entrenamiento
            sample_df = pd.DataFrame([0] * len(pipeline.features_columns)).T
            sample_df.columns = pipeline.features_columns
            res = pipeline.predict(sample_df)
            print(f"🎉 ¡Prueba de inferencia exitosa! Predicción obtenida: {res}")
            
    except Exception as e:
        print(f"ℹ️ Nota de inicialización: {e}")
        print("👉 Recuerda que para entrenar por primera vez, debes pasarle un DataFrame local.")