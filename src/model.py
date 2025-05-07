import joblib
import os
import pandas as pd
import numpy as np  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

def train_and_save_models(model_cfg, processed_dir):
    try:
        # Set MLflow tracking URI for DagsHub
        mlflow.set_tracking_uri("https://dagshub.com/gamaleldinmaya/MLops-maya.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] ="gamaleldinmaya"
        os.environ["MLFLOW_TRACKING_PASSWORD"] ="6722174301b2179cffff126f3bd479141ac066f7"

        # Start an MLflow run
        with mlflow.start_run():
            # Log parameters: Model type and learning rate (if present)
            mlflow.log_param("model_type", model_cfg['type'])
            if 'lr' in model_cfg.get('params', {}):
                mlflow.log_param("learning_rate", model_cfg['params']['lr'])

            # Load processed data
            X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
            y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
            
            # Convert y_train to 1D array to avoid warning
            y_train = np.ravel(y_train.values)  # Fix the error by importing numpy

            # Model selection and initialization
            model_classes = {
                "RandomForestClassifier": RandomForestClassifier,
                "LogisticRegression": LogisticRegression
            }

            model = model_classes[model_cfg['type']](**model_cfg.get('params', {}))

            # Train model with progress output
            print("Training model...")
            model.fit(X_train, y_train)
            print("Training completed!")

            # Save model
            os.makedirs(model_cfg['output_dir'], exist_ok=True)
            model_path = os.path.join(model_cfg['output_dir'], f"{model_cfg['name']}.pkl")
            joblib.dump(model, model_path)

            # Calculate and display training accuracy
            train_acc = model.score(X_train, y_train)
            print(f"\nModel saved to {model_path}")
            print(f"Training accuracy: {train_acc:.2%}")

            # Log metrics and the model to MLflow
            mlflow.log_metric("accuracy", train_acc)
            mlflow.sklearn.log_model(model, "model")
            
            # Return model path
            return model_path
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise RuntimeError(f"Model training failed: {str(e)}")
