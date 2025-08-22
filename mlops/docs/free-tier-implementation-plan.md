# MLOps Free Tier Implementation Plan - Predictive Maintenance System

## Project Overview

Implementation plan for MLOps pipeline using AWS Free Tier resources and local tools. This approach focuses on EC2-based model training and local model management instead of AWS SageMaker to stay within free tier limits while maintaining ML lifecycle management.

## Technology Stack (Free Tier)

- **ML Platform**: Local TensorFlow/Keras training
- **Model Storage**: S3 Free tier (5GB storage) + Local storage
- **Experiment Tracking**: MLflow (self-hosted on EC2)
- **Workflow Orchestration**: Apache Airflow (local/EC2)
- **Model Registry**: MLflow Model Registry
- **Infrastructure**: EC2 t2.micro instances
- **Monitoring**: Basic CloudWatch + Custom metrics

## Free Tier Considerations

- **EC2**: Using t2.micro instances (1 vCPU, 1GB RAM)
- **S3**: 5GB storage, 20K GET requests, 2K PUT requests
- **No SageMaker**: Using local training to avoid costs
- **Cost**: $0/month for 12 months

---

## Phase 1: Local MLOps Environment Setup

**Duration**: 3-4 days  
**Priority**: Critical

### Objectives

- Set up local MLOps environment
- Configure MLflow for experiment tracking
- Establish model training pipeline

### Tasks

1. **MLflow Setup (Local)**

   ```bash
   # Install MLflow
   pip install mlflow==2.7.1
   pip install boto3  # For S3 artifact storage

   # Create MLflow directory structure
   mkdir -p mlops/{experiments,models,data,scripts,config}
   cd mlops
   ```

2. **MLflow Configuration**

   ```python
   # mlops/config/mlflow_config.py
   import os
   import mlflow
   from mlflow.tracking import MlflowClient

   class MLflowConfig:
       def __init__(self):
           # Use local SQLite for free tier
           self.tracking_uri = "sqlite:///mlflow.db"
           self.artifact_root = "./mlartifacts"  # Local artifacts

           # S3 configuration for model storage (free tier)
           self.s3_bucket = os.getenv("S3_BUCKET", "")
           self.aws_region = os.getenv("AWS_REGION", "us-east-1")

       def setup_mlflow(self):
           """Configure MLflow for free tier usage."""
           mlflow.set_tracking_uri(self.tracking_uri)

           # Create default experiment
           try:
               experiment_id = mlflow.create_experiment(
                   "predictive-maintenance-rul",
                   artifact_location=self.artifact_root
               )
           except Exception:
               experiment_id = mlflow.get_experiment_by_name(
                   "predictive-maintenance-rul"
               ).experiment_id

           mlflow.set_experiment(experiment_id=experiment_id)
           return experiment_id

   # Initialize configuration
   config = MLflowConfig()
   ```

3. **Project Structure (Free Tier)**

   ```
   mlops/
   ├── config/
   │   ├── mlflow_config.py      # MLflow configuration
   │   └── training_config.py    # Training parameters
   ├── data/
   │   ├── raw/                  # Raw C-MAPSS data
   │   ├── processed/            # Preprocessed data
   │   └── features/             # Engineered features
   ├── experiments/
   │   ├── data_preprocessing.py # Data preparation
   │   ├── model_training.py     # Training scripts
   │   └── model_evaluation.py   # Evaluation scripts
   ├── models/
   │   ├── lstm_rul.py          # LSTM model definition
   │   └── baseline.py          # Baseline models
   ├── scripts/
   │   ├── train_local.py       # Local training script
   │   ├── evaluate_model.py    # Model evaluation
   │   └── deploy_model.py      # Model deployment
   ├── notebooks/               # Jupyter notebooks
   ├── mlflow.db               # Local MLflow database
   ├── mlartifacts/            # Local artifact storage
   └── requirements.txt
   ```

4. **Training Configuration**

   ```python
   # mlops/config/training_config.py
   from dataclasses import dataclass
   from typing import List, Dict, Any

   @dataclass
   class TrainingConfig:
       # Model parameters (optimized for t2.micro)
       sequence_length: int = 50
       batch_size: int = 32  # Small batch for memory efficiency
       epochs: int = 50
       learning_rate: float = 0.001

       # Architecture parameters
       lstm_units: int = 32  # Reduced for free tier
       dense_units: int = 16
       dropout_rate: float = 0.2

       # Data parameters
       train_ratio: float = 0.7
       val_ratio: float = 0.2
       test_ratio: float = 0.1

       # Free tier optimizations
       early_stopping_patience: int = 10
       reduce_lr_patience: int = 5
       max_training_time_minutes: int = 30  # Limit for t2.micro

       # Paths
       data_path: str = "data/processed"
       model_output_path: str = "models"

       def to_dict(self) -> Dict[str, Any]:
           return {
               'sequence_length': self.sequence_length,
               'batch_size': self.batch_size,
               'epochs': self.epochs,
               'learning_rate': self.learning_rate,
               'lstm_units': self.lstm_units,
               'dense_units': self.dense_units,
               'dropout_rate': self.dropout_rate
           }
   ```

### Deliverables

- [ ] MLflow environment configured locally
- [ ] Project structure established
- [ ] Training configuration optimized for free tier

---

## Phase 2: Data Processing Pipeline (Free Tier)

**Duration**: 4-5 days  
**Priority**: High

### Objectives

- Implement data preprocessing for C-MAPSS dataset
- Create feature engineering pipeline
- Optimize for limited memory usage

### Tasks

1. **Data Preprocessing Script**

   ```python
   # mlops/experiments/data_preprocessing.py
   import pandas as pd
   import numpy as np
   import mlflow
   import logging
   from sklearn.preprocessing import MinMaxScaler
   from typing import Tuple, Dict

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   class DataPreprocessor:
       def __init__(self, config):
           self.config = config
           self.scaler = MinMaxScaler()

       def load_cmapss_data(self, file_path: str) -> pd.DataFrame:
           """Load C-MAPSS dataset with memory optimization."""

           # Column names for C-MAPSS dataset
           columns = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                    [f'sensor_{i}' for i in range(1, 22)]

           # Load with dtype optimization for memory
           df = pd.read_csv(
               file_path,
               sep=' ',
               header=None,
               names=columns,
               dtype=np.float32
           )

           return df

       def calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
           """Calculate Remaining Useful Life for each engine."""

           # Calculate max cycle for each unit (failure point)
           max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
           max_cycles.columns = ['unit_id', 'max_cycle']

           # Merge and calculate RUL
           df = df.merge(max_cycles, on='unit_id')
           df['RUL'] = df['max_cycle'] - df['cycle']

           # Cap RUL at reasonable maximum for better training
           df['RUL'] = np.minimum(df['RUL'], 125)

           return df.drop('max_cycle', axis=1)

       def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
           """Create additional features (lightweight for free tier)."""

           # Rolling averages (memory efficient)
           sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

           for col in sensor_cols:
               # 5-cycle rolling average
               df[f'{col}_ma5'] = df.groupby('unit_id')[col].rolling(
                   window=5, min_periods=1
               ).mean().reset_index(0, drop=True)

           # Basic statistical features
           df['sensor_mean'] = df[sensor_cols].mean(axis=1)
           df['sensor_std'] = df[sensor_cols].std(axis=1)

           return df

       def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
           """Create sequences for LSTM training."""

           X, y = [], []

           for unit_id in df['unit_id'].unique():
               unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')

               # Feature columns (exclude metadata)
               feature_cols = [col for col in unit_data.columns
                             if col not in ['unit_id', 'cycle', 'RUL']]

               features = unit_data[feature_cols].values
               rul_values = unit_data['RUL'].values

               # Create sequences
               for i in range(len(features) - self.config.sequence_length + 1):
                   X.append(features[i:i + self.config.sequence_length])
                   y.append(rul_values[i + self.config.sequence_length - 1])

           return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

       def preprocess_dataset(self, train_path: str, test_path: str) -> Dict:
           """Complete preprocessing pipeline with MLflow tracking."""

           with mlflow.start_run(run_name="data_preprocessing"):
               logger.info("Starting data preprocessing...")

               # Load data
               train_df = self.load_cmapss_data(train_path)
               test_df = self.load_cmapss_data(test_path)

               logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

               # Calculate RUL
               train_df = self.calculate_rul(train_df)
               test_df = self.calculate_rul(test_df)

               # Feature engineering
               train_df = self.feature_engineering(train_df)
               test_df = self.feature_engineering(test_df)

               # Create sequences
               X_train, y_train = self.create_sequences(train_df)
               X_test, y_test = self.create_sequences(test_df)

               # Normalize features
               X_train_flat = X_train.reshape(-1, X_train.shape[-1])
               X_train_scaled = self.scaler.fit_transform(X_train_flat)
               X_train = X_train_scaled.reshape(X_train.shape)

               X_test_flat = X_test.reshape(-1, X_test.shape[-1])
               X_test_scaled = self.scaler.transform(X_test_flat)
               X_test = X_test_scaled.reshape(X_test.shape)

               # Log metrics
               mlflow.log_param("sequence_length", self.config.sequence_length)
               mlflow.log_param("num_features", X_train.shape[-1])
               mlflow.log_metric("train_samples", len(X_train))
               mlflow.log_metric("test_samples", len(X_test))

               # Save processed data locally (free tier)
               processed_data = {
                   'X_train': X_train,
                   'y_train': y_train,
                   'X_test': X_test,
                   'y_test': y_test,
                   'scaler': self.scaler
               }

               # Save to local storage
               import pickle
               with open('data/processed/processed_data.pkl', 'wb') as f:
                   pickle.dump(processed_data, f)

               logger.info("Data preprocessing completed!")
               return processed_data
   ```

2. **Data Loading Utilities**

   ```python
   # mlops/experiments/data_utils.py
   import pickle
   import numpy as np
   from typing import Tuple, Optional

   def load_processed_data(data_path: str = "data/processed/processed_data.pkl") -> dict:
       """Load preprocessed data from local storage."""
       try:
           with open(data_path, 'rb') as f:
               return pickle.load(f)
       except FileNotFoundError:
           raise FileNotFoundError(
               f"Processed data not found at {data_path}. "
               "Run data preprocessing first."
           )

   def create_data_splits(
       X: np.ndarray,
       y: np.ndarray,
       train_ratio: float = 0.7,
       val_ratio: float = 0.2
   ) -> Tuple[np.ndarray, ...]:
       """Split data into train/validation/test sets."""

       n_samples = len(X)
       n_train = int(n_samples * train_ratio)
       n_val = int(n_samples * val_ratio)

       # Random shuffle
       indices = np.random.permutation(n_samples)

       train_idx = indices[:n_train]
       val_idx = indices[n_train:n_train + n_val]
       test_idx = indices[n_train + n_val:]

       return (
           X[train_idx], y[train_idx],
           X[val_idx], y[val_idx],
           X[test_idx], y[test_idx]
       )
   ```

### Deliverables

- [ ] Data preprocessing pipeline implemented
- [ ] Feature engineering optimized for memory
- [ ] Data loading utilities created

---

## Phase 3: Local Model Training Pipeline

**Duration**: 5-6 days  
**Priority**: High

### Objectives

- Implement LSTM model for RUL prediction
- Create training pipeline with MLflow tracking
- Optimize for t2.micro constraints

### Tasks

1. **LSTM Model Definition**

   ```python
   # mlops/models/lstm_rul.py
   import tensorflow as tf
   import numpy as np
   from typing import Tuple
   import logging

   logger = logging.getLogger(__name__)

   class LSTMRULModel:
       """LSTM model for RUL prediction optimized for free tier."""

       def __init__(self, config):
           self.config = config
           self.model = None

       def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
           """Build LSTM model with memory optimization."""

           model = tf.keras.Sequential([
               # Input layer
               tf.keras.layers.Input(shape=input_shape),

               # LSTM layers (reduced for t2.micro)
               tf.keras.layers.LSTM(
                   self.config.lstm_units,
                   return_sequences=True,
                   dropout=self.config.dropout_rate
               ),
               tf.keras.layers.LSTM(
                   self.config.lstm_units // 2,
                   dropout=self.config.dropout_rate
               ),

               # Dense layers
               tf.keras.layers.Dense(
                   self.config.dense_units,
                   activation='relu'
               ),
               tf.keras.layers.Dropout(self.config.dropout_rate),
               tf.keras.layers.Dense(1, activation='linear')
           ])

           # Compile with memory-efficient optimizer
           model.compile(
               optimizer=tf.keras.optimizers.Adam(
                   learning_rate=self.config.learning_rate
               ),
               loss='mse',
               metrics=['mae', 'mape']
           )

           self.model = model
           return model

       def get_callbacks(self):
           """Get training callbacks optimized for free tier."""

           callbacks = [
               # Early stopping to prevent overfitting
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=self.config.early_stopping_patience,
                   restore_best_weights=True
               ),

               # Reduce learning rate on plateau
               tf.keras.callbacks.ReduceLROnPlateau(
                   monitor='val_loss',
                   factor=0.5,
                   patience=self.config.reduce_lr_patience,
                   min_lr=1e-6
               ),

               # Memory cleanup
               tf.keras.callbacks.LambdaCallback(
                   on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session()
               )
           ]

           return callbacks
   ```

2. **Training Script with MLflow**

   ```python
   # mlops/scripts/train_local.py
   import mlflow
   import mlflow.tensorflow
   import tensorflow as tf
   import numpy as np
   import logging
   import pickle
   import time
   from pathlib import Path

   from ..config.mlflow_config import config as mlflow_config
   from ..config.training_config import TrainingConfig
   from ..models.lstm_rul import LSTMRULModel
   from ..experiments.data_utils import load_processed_data, create_data_splits

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def train_model():
       """Train LSTM model with MLflow tracking."""

       # Setup MLflow
       mlflow_config.setup_mlflow()

       # Load configuration
       config = TrainingConfig()

       with mlflow.start_run(run_name="lstm_rul_training"):
           try:
               # Log parameters
               mlflow.log_params(config.to_dict())

               # Load processed data
               logger.info("Loading processed data...")
               data = load_processed_data()

               X_train = data['X_train']
               y_train = data['y_train']

               # Create validation split
               X_train, y_train, X_val, y_val, X_test, y_test = create_data_splits(
                   X_train, y_train,
                   config.train_ratio,
                   config.val_ratio
               )

               logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

               # Build model
               model_builder = LSTMRULModel(config)
               model = model_builder.build_model(
                   input_shape=(config.sequence_length, X_train.shape[-1])
               )

               logger.info(f"Model built with {model.count_params()} parameters")
               mlflow.log_metric("model_parameters", model.count_params())

               # Train model
               start_time = time.time()

               history = model.fit(
                   X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=config.epochs,
                   batch_size=config.batch_size,
                   callbacks=model_builder.get_callbacks(),
                   verbose=1
               )

               training_time = time.time() - start_time
               mlflow.log_metric("training_time_minutes", training_time / 60)

               # Evaluate on test set
               test_loss, test_mae, test_mape = model.evaluate(X_test, y_test, verbose=0)

               mlflow.log_metric("test_loss", test_loss)
               mlflow.log_metric("test_mae", test_mae)
               mlflow.log_metric("test_mape", test_mape)

               # Log training history
               for epoch, (loss, val_loss) in enumerate(zip(
                   history.history['loss'],
                   history.history['val_loss']
               )):
                   mlflow.log_metric("train_loss", loss, step=epoch)
                   mlflow.log_metric("val_loss", val_loss, step=epoch)

               # Save model locally
               model_path = Path(config.model_output_path)
               model_path.mkdir(exist_ok=True)

               model.save(model_path / "lstm_rul_model.h5")

               # Save scaler
               with open(model_path / "scaler.pkl", 'wb') as f:
                   pickle.dump(data['scaler'], f)

               # Log model with MLflow
               mlflow.tensorflow.log_model(
                   model,
                   "model",
                   registered_model_name="lstm_rul_predictor"
               )

               logger.info("Training completed successfully!")

               return {
                   'model': model,
                   'test_mae': test_mae,
                   'test_mape': test_mape,
                   'training_time': training_time
               }

           except Exception as e:
               logger.error(f"Training failed: {e}")
               mlflow.log_param("error", str(e))
               raise

   if __name__ == "__main__":
       train_model()
   ```

3. **Model Evaluation Script**

   ```python
   # mlops/scripts/evaluate_model.py
   import mlflow
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
   import logging

   logger = logging.getLogger(__name__)

   def evaluate_model(model_path: str, test_data_path: str):
       """Comprehensive model evaluation with MLflow logging."""

       with mlflow.start_run(run_name="model_evaluation"):
           # Load model and data
           model = tf.keras.models.load_model(model_path)
           data = load_processed_data(test_data_path)

           X_test = data['X_test']
           y_test = data['y_test']

           # Make predictions
           y_pred = model.predict(X_test).flatten()

           # Calculate metrics
           mae = mean_absolute_error(y_test, y_pred)
           mse = mean_squared_error(y_test, y_pred)
           rmse = np.sqrt(mse)
           r2 = r2_score(y_test, y_pred)

           # Log metrics
           mlflow.log_metric("eval_mae", mae)
           mlflow.log_metric("eval_mse", mse)
           mlflow.log_metric("eval_rmse", rmse)
           mlflow.log_metric("eval_r2", r2)

           # Create evaluation plots (memory efficient)
           fig, axes = plt.subplots(2, 2, figsize=(12, 10))

           # Prediction vs Actual
           axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
           axes[0, 0].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()], 'r--')
           axes[0, 0].set_xlabel('Actual RUL')
           axes[0, 0].set_ylabel('Predicted RUL')
           axes[0, 0].set_title('Prediction vs Actual')

           # Residuals
           residuals = y_test - y_pred
           axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
           axes[0, 1].axhline(y=0, color='r', linestyle='--')
           axes[0, 1].set_xlabel('Predicted RUL')
           axes[0, 1].set_ylabel('Residuals')
           axes[0, 1].set_title('Residual Plot')

           # Error distribution
           axes[1, 0].hist(residuals, bins=50, alpha=0.7)
           axes[1, 0].set_xlabel('Residuals')
           axes[1, 0].set_ylabel('Frequency')
           axes[1, 0].set_title('Error Distribution')

           # RUL distribution
           axes[1, 1].hist(y_test, bins=50, alpha=0.7, label='Actual')
           axes[1, 1].hist(y_pred, bins=50, alpha=0.7, label='Predicted')
           axes[1, 1].set_xlabel('RUL')
           axes[1, 1].set_ylabel('Frequency')
           axes[1, 1].set_title('RUL Distribution')
           axes[1, 1].legend()

           plt.tight_layout()
           plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
           mlflow.log_artifact('evaluation_plots.png')
           plt.close()

           logger.info(f"Evaluation completed - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

           return {
               'mae': mae,
               'rmse': rmse,
               'r2': r2,
               'predictions': y_pred
           }
   ```

### Deliverables

- [ ] LSTM model implemented and optimized
- [ ] Training pipeline with MLflow tracking
- [ ] Model evaluation scripts created

---

## Phase 4: Model Deployment Pipeline (Free Tier)

**Duration**: 3-4 days  
**Priority**: Medium

### Objectives

- Create model deployment scripts
- Implement model serving pipeline
- Set up model monitoring

### Tasks

1. **Model Deployment Script**

   ```python
   # mlops/scripts/deploy_model.py
   import mlflow
   import shutil
   import boto3
   import logging
   from pathlib import Path

   logger = logging.getLogger(__name__)

   def deploy_model_to_s3(model_path: str, s3_bucket: str, model_name: str):
       """Deploy model to S3 for free tier serving."""

       try:
           # Create deployment package
           deployment_path = Path("deployment_package")
           deployment_path.mkdir(exist_ok=True)

           # Copy model files
           shutil.copy(f"{model_path}/lstm_rul_model.h5", deployment_path)
           shutil.copy(f"{model_path}/scaler.pkl", deployment_path)

           # Create model metadata
           model_info = {
               "model_name": model_name,
               "model_version": "1.0.0",
               "framework": "tensorflow",
               "input_shape": [50, 30],  # Example shape
               "output_shape": [1]
           }

           import json
           with open(deployment_path / "model_info.json", 'w') as f:
               json.dump(model_info, f)

           # Upload to S3
           s3_client = boto3.client('s3')

           for file_path in deployment_path.iterdir():
               if file_path.is_file():
                   s3_key = f"models/{model_name}/{file_path.name}"
                   s3_client.upload_file(
                       str(file_path),
                       s3_bucket,
                       s3_key
                   )
                   logger.info(f"Uploaded {file_path.name} to s3://{s3_bucket}/{s3_key}")

           # Cleanup
           shutil.rmtree(deployment_path)

           logger.info(f"Model {model_name} deployed successfully to S3")

       except Exception as e:
           logger.error(f"Deployment failed: {e}")
           raise

   def register_model_in_mlflow(model_path: str, model_name: str):
       """Register model in MLflow model registry."""

       with mlflow.start_run(run_name="model_registration"):
           # Log the model
           model_uri = mlflow.tensorflow.log_model(
               tf.keras.models.load_model(model_path),
               "model",
               registered_model_name=model_name
           )

           # Transition to production
           client = mlflow.tracking.MlflowClient()
           latest_version = client.get_latest_versions(
               model_name,
               stages=["None"]
           )[0]

           client.transition_model_version_stage(
               name=model_name,
               version=latest_version.version,
               stage="Production"
           )

           logger.info(f"Model {model_name} registered and moved to Production")

           return model_uri
   ```

2. **Model Monitoring Setup**

   ```python
   # mlops/scripts/monitor_model.py
   import time
   import psutil
   import logging
   from typing import Dict, Any
   import boto3
   import json

   logger = logging.getLogger(__name__)

   class ModelMonitor:
       """Simple model monitoring for free tier."""

       def __init__(self, model_name: str):
           self.model_name = model_name
           self.metrics = {}

       def log_prediction_metrics(self,
                                prediction_time: float,
                                input_size: int,
                                confidence: float = None):
           """Log prediction metrics."""

           timestamp = time.time()

           self.metrics[timestamp] = {
               "model_name": self.model_name,
               "prediction_time_ms": prediction_time * 1000,
               "input_size": input_size,
               "confidence": confidence,
               "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
               "cpu_percent": psutil.cpu_percent()
           }

           # Log to CloudWatch (free tier)
           try:
               cloudwatch = boto3.client('cloudwatch')

               cloudwatch.put_metric_data(
                   Namespace='MLOps/ModelServing',
                   MetricData=[
                       {
                           'MetricName': 'PredictionLatency',
                           'Value': prediction_time * 1000,
                           'Unit': 'Milliseconds',
                           'Dimensions': [
                               {
                                   'Name': 'ModelName',
                                   'Value': self.model_name
                               }
                           ]
                       },
                       {
                           'MetricName': 'MemoryUsage',
                           'Value': psutil.Process().memory_info().rss / 1024 / 1024,
                           'Unit': 'Megabytes',
                           'Dimensions': [
                               {
                                   'Name': 'ModelName',
                                   'Value': self.model_name
                               }
                           ]
                       }
                   ]
               )

           except Exception as e:
               logger.warning(f"Failed to send metrics to CloudWatch: {e}")

       def get_model_health(self) -> Dict[str, Any]:
           """Get current model health status."""

           recent_metrics = list(self.metrics.values())[-10:]  # Last 10 predictions

           if not recent_metrics:
               return {"status": "no_data"}

           avg_latency = sum(m["prediction_time_ms"] for m in recent_metrics) / len(recent_metrics)
           avg_memory = sum(m["memory_usage_mb"] for m in recent_metrics) / len(recent_metrics)

           status = "healthy"
           if avg_latency > 2000:  # 2 seconds
               status = "degraded"
           if avg_memory > 700:  # 700MB for t2.micro
               status = "memory_warning"

           return {
               "status": status,
               "avg_latency_ms": avg_latency,
               "avg_memory_mb": avg_memory,
               "total_predictions": len(self.metrics)
           }
   ```

### Deliverables

- [ ] Model deployment scripts created
- [ ] S3 deployment pipeline implemented
- [ ] Basic model monitoring setup

---

## Migration Strategy to Paid Services (SageMaker)

### Phase 1: Preparation (Month 10-11)

1. **SageMaker Migration Planning**

   ```python
   # mlops/migration/sagemaker_migration.py
   import boto3
   import sagemaker
   from sagemaker.tensorflow import TensorFlow

   def prepare_sagemaker_migration():
       """Prepare for migration to SageMaker."""

       # Create SageMaker session
       sagemaker_session = sagemaker.Session()

       # Convert training script for SageMaker
       training_script = """
       import tensorflow as tf
       import argparse
       import os

       def train():
           # SageMaker training code
           pass

       if __name__ == '__main__':
           train()
       """

       # Create SageMaker estimator
       estimator = TensorFlow(
           entry_point='train.py',
           role='arn:aws:iam::account:role/SageMakerRole',
           instance_type='ml.m5.large',
           instance_count=1,
           framework_version='2.11',
           py_version='py39'
       )

       return estimator
   ```

2. **Pipeline Migration**

   ```python
   # Convert local pipeline to SageMaker Pipelines
   from sagemaker.workflow.pipeline import Pipeline
   from sagemaker.workflow.steps import TrainingStep

   def create_sagemaker_pipeline():
       """Create SageMaker pipeline from local workflow."""

       # Define training step
       training_step = TrainingStep(
           name="training-step",
           estimator=estimator,
           inputs={
               "training": "s3://bucket/training-data",
               "validation": "s3://bucket/validation-data"
           }
       )

       # Create pipeline
       pipeline = Pipeline(
           name="rul-prediction-pipeline",
           steps=[training_step]
       )

       return pipeline
   ```

### Phase 2: Migration Execution (Month 12)

1. **Data Migration to S3**
2. **Model Training on SageMaker**
3. **Endpoint Deployment**
4. **Pipeline Automation**

---

## Success Metrics (Free Tier)

### Training Performance

- **Training Time**: < 30 minutes (t2.micro constraint)
- **Model Accuracy**: MAE < 15 RUL cycles
- **Memory Usage**: < 700MB during training
- **Storage Usage**: < 2GB (within S3 free tier)

### Operational Metrics

- **Experiment Tracking**: All runs logged in MLflow
- **Model Versioning**: Automated model registry
- **Deployment Time**: < 5 minutes
- **Cost**: $0/month (within free tier)

---

## Timeline Summary

| Phase                     | Duration | Dependencies       |
| ------------------------- | -------- | ------------------ |
| Phase 1: MLOps Setup      | 3-4 days | Python Environment |
| Phase 2: Data Processing  | 4-5 days | Phase 1            |
| Phase 3: Model Training   | 5-6 days | Phase 2            |
| Phase 4: Model Deployment | 3-4 days | Phase 3            |

**Total Duration**: 15-19 days (3-4 weeks)

---

## Next Steps

1. **Immediate**: Set up local MLflow environment
2. **Week 1**: Implement data processing pipeline
3. **Week 2-3**: Develop and train LSTM model
4. **Week 4**: Set up deployment and monitoring
5. **Month 6-10**: Monitor and optimize models
6. **Month 12**: Execute migration to SageMaker
