# Python ML Service Implementation Plan - Predictive Maintenance System

## Project Overview
Implementation plan for the Python ML microservice that handles machine learning model serving for RUL (Remaining Useful Life) predictions. This service manages LSTM model inference, communicates via gRPC with the API Gateway, and interfaces with AWS SageMaker for model deployment.

## Technology Stack
- **Language**: Python 3.9+
- **Web Framework**: FastAPI
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Communication**: gRPC (grpcio)
- **Model Serving**: AWS SageMaker Endpoint
- **Testing**: pytest, pytest-asyncio
- **Deployment**: Docker, AWS Fargate/EC2

---

## Phase 1: Project Setup and Foundation
**Duration**: 2-3 days  
**Priority**: Critical

### Objectives
- Initialize Python project with proper structure
- Set up development environment
- Configure dependencies and tooling

### Tasks
1. **Project Initialization**
   - [ ] Create project structure
   - [ ] Set up virtual environment
   - [ ] Configure pyproject.toml or setup.py
   - [ ] Set up .gitignore
   - [ ] Configure pre-commit hooks

2. **Dependencies Installation**
   ```toml
   # pyproject.toml
   [tool.poetry.dependencies]
   python = "^3.9"
   fastapi = "^0.104.0"
   grpcio = "^1.59.0"
   grpcio-tools = "^1.59.0"
   tensorflow = "^2.14.0"
   numpy = "^1.24.0"
   pandas = "^2.1.0"
   scikit-learn = "^1.3.0"
   boto3 = "^1.28.0"
   pydantic = "^2.4.0"
   python-dotenv = "^1.0.0"
   structlog = "^23.1.0"
   
   [tool.poetry.dev-dependencies]
   pytest = "^7.4.0"
   pytest-asyncio = "^0.21.0"
   pytest-cov = "^4.1.0"
   black = "^23.9.0"
   isort = "^5.12.0"
   mypy = "^1.5.0"
   flake8 = "^6.1.0"
   ```

3. **Project Structure**
   ```
   backend/ml-service/
   ├── src/
   │   ├── api/                    # FastAPI endpoints
   │   ├── grpc_server/           # gRPC server implementation
   │   ├── models/                # ML model classes
   │   ├── preprocessing/         # Data preprocessing
   │   ├── inference/             # Inference logic
   │   ├── sagemaker/             # SageMaker integration
   │   ├── config/                # Configuration
   │   ├── utils/                 # Utilities
   │   └── main.py
   ├── proto/                      # Protocol buffer definitions
   ├── notebooks/                  # Jupyter notebooks for analysis
   ├── tests/
   │   ├── unit/
   │   ├── integration/
   │   └── fixtures/
   ├── models/                     # Trained model artifacts
   ├── data/                       # Local data for testing
   └── docker/
   ```

4. **Configuration Management**
   ```python
   # src/config/settings.py
   from pydantic_settings import BaseSettings
   
   class Settings(BaseSettings):
       # Service Configuration
       service_name: str = "ml-prediction-service"
       grpc_port: int = 50051
       http_port: int = 8000
       
       # Model Configuration
       model_path: str = "./models/lstm_model.h5"
       sequence_length: int = 50
       n_features: int = 14
       rul_ceiling: int = 125
       
       # SageMaker Configuration
       sagemaker_endpoint: str = ""
       aws_region: str = "us-east-1"
       
       # Supabase Configuration
       supabase_url: str = ""
       supabase_key: str = ""
       
       # Logging
       log_level: str = "INFO"
       
       class Config:
           env_file = ".env"
   
   settings = Settings()
   ```

### Deliverables
- Initialized Python project
- Configured development environment
- Dependency management setup

### Testing Checklist
- [ ] Virtual environment activates
- [ ] All dependencies install
- [ ] Linting tools work
- [ ] Project structure created

---

## Phase 2: Data Processing Pipeline
**Duration**: 4-5 days  
**Priority**: Critical

### Objectives
- Implement data preprocessing pipeline
- Create feature engineering module
- Handle C-MAPSS dataset specifics

### Tasks
1. **Data Loading Module**
   ```python
   # src/preprocessing/data_loader.py
   import pandas as pd
   import numpy as np
   from typing import Tuple, List
   
   class CMAPSSDataLoader:
       """Loader for NASA C-MAPSS Turbofan Engine Dataset"""
       
       def __init__(self, data_path: str):
           self.data_path = data_path
           self.sensor_columns = ['s2', 's3', 's4', 's7', 's8', 's9', 
                                  's11', 's12', 's13', 's14', 's15', 
                                  's17', 's20', 's21']
           self.drop_columns = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
           
       def load_train_data(self) -> pd.DataFrame:
           """Load and parse training data from FD001"""
           columns = ['unit_number', 'time_cycles'] + \
                    ['setting_1', 'setting_2', 'setting_3'] + \
                    [f's{i}' for i in range(1, 22)]
           
           df = pd.read_csv(
               f"{self.data_path}/train_FD001.txt",
               sep=' ',
               header=None,
               names=columns,
               index_col=False
           )
           
           # Drop non-informative sensors
           df = df.drop(columns=self.drop_columns)
           
           # Add RUL labels
           df = self._add_rul_labels(df)
           
           return df
       
       def _add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
           """Calculate RUL for each cycle"""
           max_cycles = df.groupby('unit_number')['time_cycles'].max()
           
           def calculate_rul(row):
               max_cycle = max_cycles[row['unit_number']]
               rul = max_cycle - row['time_cycles']
               return min(rul, self.rul_ceiling)  # Apply ceiling
           
           df['RUL'] = df.apply(calculate_rul, axis=1)
           return df
   ```

2. **Feature Engineering**
   ```python
   # src/preprocessing/feature_engineering.py
   from sklearn.preprocessing import MinMaxScaler
   import numpy as np
   
   class FeatureEngineer:
       def __init__(self, sequence_length: int = 50):
           self.sequence_length = sequence_length
           self.scaler = MinMaxScaler(feature_range=(0, 1))
           self.feature_columns = None
           
       def fit(self, df: pd.DataFrame):
           """Fit the scaler on training data"""
           self.feature_columns = [col for col in df.columns 
                                  if col.startswith('s')]
           self.scaler.fit(df[self.feature_columns])
           
       def transform(self, df: pd.DataFrame) -> np.ndarray:
           """Transform and normalize sensor data"""
           normalized = self.scaler.transform(df[self.feature_columns])
           return normalized
       
       def create_sequences(self, data: np.ndarray, 
                          labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
           """Create sliding window sequences for LSTM input"""
           sequences = []
           targets = []
           
           for i in range(len(data) - self.sequence_length + 1):
               sequences.append(data[i:i + self.sequence_length])
               if labels is not None:
                   targets.append(labels[i + self.sequence_length - 1])
           
           return np.array(sequences), np.array(targets) if labels is not None else None
       
       def preprocess_for_inference(self, sensor_data: List[dict]) -> np.ndarray:
           """Preprocess raw sensor data for model inference"""
           df = pd.DataFrame(sensor_data)
           
           # Ensure we have the right columns
           df = df[self.feature_columns]
           
           # Normalize
           normalized = self.transform(df)
           
           # Create sequence
           if len(normalized) < self.sequence_length:
               # Pad with zeros if not enough data
               padding = np.zeros((self.sequence_length - len(normalized), 
                                  len(self.feature_columns)))
               normalized = np.vstack([padding, normalized])
           else:
               # Take last sequence_length readings
               normalized = normalized[-self.sequence_length:]
           
           return normalized.reshape(1, self.sequence_length, len(self.feature_columns))
   ```

3. **Data Validation**
   ```python
   # src/preprocessing/validation.py
   from pydantic import BaseModel, validator
   from typing import List, Optional
   
   class SensorReading(BaseModel):
       s2: float   # T24 - Total temperature at LPC outlet
       s3: float   # T30 - Total temperature at HPC outlet
       s4: float   # T50 - Total temperature at LPT outlet
       s7: float   # P30 - Total pressure at HPC outlet
       s8: float   # Nf - Physical fan speed
       s9: float   # Nc - Physical core speed
       s11: float  # Ps30 - Static pressure at HPC outlet
       s12: float  # phi - Ratio of fuel flow to Ps30
       s13: float  # NRf - Corrected fan speed
       s14: float  # NRc - Corrected core speed
       s15: float  # BPR - Bypass Ratio
       s17: float  # htBleed - HPT coolant bleed
       s20: float  # W31 - LPT coolant bleed
       s21: float  # W32 - HPT coolant bleed
       
       @validator('*')
       def check_sensor_range(cls, v, field):
           """Validate sensor values are within reasonable ranges"""
           if v < 0:
               raise ValueError(f"{field.name} cannot be negative")
           if v > 10000:  # Adjust based on actual sensor ranges
               raise ValueError(f"{field.name} value too high")
           return v
   
   class PredictionRequest(BaseModel):
       engine_id: str
       sensor_data: List[SensorReading]
       
       @validator('sensor_data')
       def check_sequence_length(cls, v):
           if len(v) == 0:
               raise ValueError("Sensor data cannot be empty")
           if len(v) > 500:  # Max sequence length
               raise ValueError("Too many sensor readings")
           return v
   ```

4. **Preprocessing Pipeline**
   ```python
   # src/preprocessing/pipeline.py
   import joblib
   
   class PreprocessingPipeline:
       def __init__(self, config):
           self.config = config
           self.data_loader = CMAPSSDataLoader(config.data_path)
           self.feature_engineer = FeatureEngineer(config.sequence_length)
           
       def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
           """Prepare data for model training"""
           # Load data
           df = self.data_loader.load_train_data()
           
           # Fit feature engineer
           self.feature_engineer.fit(df)
           
           # Process each engine separately
           X_sequences = []
           y_sequences = []
           
           for unit in df['unit_number'].unique():
               unit_df = df[df['unit_number'] == unit]
               
               # Transform features
               features = self.feature_engineer.transform(unit_df)
               labels = unit_df['RUL'].values
               
               # Create sequences
               X, y = self.feature_engineer.create_sequences(features, labels)
               X_sequences.append(X)
               y_sequences.append(y)
           
           return np.vstack(X_sequences), np.hstack(y_sequences)
       
       def save_preprocessor(self, path: str):
           """Save fitted preprocessor for inference"""
           joblib.dump(self.feature_engineer, f"{path}/feature_engineer.pkl")
       
       def load_preprocessor(self, path: str):
           """Load fitted preprocessor"""
           self.feature_engineer = joblib.load(f"{path}/feature_engineer.pkl")
   ```

### Deliverables
- Data loading module
- Feature engineering pipeline
- Data validation schemas
- Preprocessing utilities

### Testing Checklist
- [ ] Data loads correctly
- [ ] Feature engineering works
- [ ] Sequences created properly
- [ ] Validation catches errors

---

## Phase 3: Model Development and Training
**Duration**: 5-6 days  
**Priority**: Critical

### Objectives
- Implement LSTM model architecture
- Create training pipeline
- Implement evaluation metrics

### Tasks
1. **LSTM Model Architecture**
   ```python
   # src/models/lstm_model.py
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers
   from typing import Tuple
   
   class RULPredictionModel:
       def __init__(self, config):
           self.config = config
           self.model = None
           self.history = None
           
       def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
           """Build LSTM architecture for RUL prediction"""
           model = keras.Sequential([
               # Input layer
               layers.Input(shape=input_shape),
               
               # First LSTM layer
               layers.LSTM(
                   units=64,
                   return_sequences=True,
                   kernel_regularizer=keras.regularizers.l2(0.001)
               ),
               layers.Dropout(0.2),
               
               # Second LSTM layer
               layers.LSTM(
                   units=64,
                   return_sequences=False,
                   kernel_regularizer=keras.regularizers.l2(0.001)
               ),
               layers.Dropout(0.2),
               
               # Dense layers
               layers.Dense(32, activation='relu'),
               layers.Dropout(0.2),
               
               # Output layer
               layers.Dense(1, activation='linear')
           ])
           
           # Compile model
           model.compile(
               optimizer=keras.optimizers.Adam(learning_rate=0.001),
               loss='mse',
               metrics=['mae', 'mape']
           )
           
           self.model = model
           return model
       
       def train(self, X_train, y_train, X_val, y_val, 
                epochs=100, batch_size=256):
           """Train the LSTM model"""
           
           # Callbacks
           callbacks = [
               keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=10,
                   restore_best_weights=True
               ),
               keras.callbacks.ReduceLROnPlateau(
                   monitor='val_loss',
                   factor=0.5,
                   patience=5,
                   min_lr=1e-7
               ),
               keras.callbacks.ModelCheckpoint(
                   filepath=f"{self.config.model_path}/best_model.h5",
                   monitor='val_loss',
                   save_best_only=True
               )
           ]
           
           # Train model
           self.history = self.model.fit(
               X_train, y_train,
               validation_data=(X_val, y_val),
               epochs=epochs,
               batch_size=batch_size,
               callbacks=callbacks,
               verbose=1
           )
           
           return self.history
       
       def predict(self, X: np.ndarray) -> np.ndarray:
           """Make predictions"""
           predictions = self.model.predict(X)
           # Ensure predictions are non-negative
           return np.maximum(predictions, 0)
       
       def save_model(self, path: str):
           """Save trained model"""
           self.model.save(f"{path}/lstm_rul_model.h5")
           
       def load_model(self, path: str):
           """Load trained model"""
           self.model = keras.models.load_model(f"{path}/lstm_rul_model.h5")
   ```

2. **Model Evaluation**
   ```python
   # src/models/evaluation.py
   import numpy as np
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   
   class ModelEvaluator:
       def __init__(self):
           self.metrics = {}
       
       def calculate_rmse(self, y_true, y_pred):
           """Calculate Root Mean Squared Error"""
           return np.sqrt(mean_squared_error(y_true, y_pred))
       
       def calculate_mae(self, y_true, y_pred):
           """Calculate Mean Absolute Error"""
           return mean_absolute_error(y_true, y_pred)
       
       def calculate_phm08_score(self, y_true, y_pred):
           """
           Calculate PHM08 scoring function
           Penalizes late predictions more than early predictions
           """
           d = y_pred - y_true
           score = np.sum(np.where(d < 0, 
                                   np.exp(-d/13) - 1,
                                   np.exp(d/10) - 1))
           return score
       
       def evaluate(self, model, X_test, y_test):
           """Comprehensive model evaluation"""
           y_pred = model.predict(X_test).flatten()
           
           self.metrics = {
               'rmse': self.calculate_rmse(y_test, y_pred),
               'mae': self.calculate_mae(y_test, y_pred),
               'phm08_score': self.calculate_phm08_score(y_test, y_pred),
               'mean_prediction': np.mean(y_pred),
               'std_prediction': np.std(y_pred)
           }
           
           return self.metrics
       
       def plot_predictions(self, y_true, y_pred, save_path=None):
           """Plot predictions vs actual values"""
           import matplotlib.pyplot as plt
           
           plt.figure(figsize=(12, 6))
           
           # Scatter plot
           plt.subplot(1, 2, 1)
           plt.scatter(y_true, y_pred, alpha=0.5)
           plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
           plt.xlabel('Actual RUL')
           plt.ylabel('Predicted RUL')
           plt.title('Predictions vs Actual')
           
           # Error distribution
           plt.subplot(1, 2, 2)
           errors = y_pred - y_true
           plt.hist(errors, bins=50, edgecolor='black')
           plt.xlabel('Prediction Error')
           plt.ylabel('Frequency')
           plt.title('Error Distribution')
           
           if save_path:
               plt.savefig(save_path)
           plt.show()
   ```

3. **Training Pipeline**
   ```python
   # src/models/training_pipeline.py
   from sklearn.model_selection import train_test_split
   import mlflow
   import mlflow.tensorflow
   
   class TrainingPipeline:
       def __init__(self, config):
           self.config = config
           self.preprocessor = PreprocessingPipeline(config)
           self.model = RULPredictionModel(config)
           self.evaluator = ModelEvaluator()
           
       def run_training(self):
           """Execute complete training pipeline"""
           
           # Start MLflow run
           with mlflow.start_run():
               # Log parameters
               mlflow.log_params({
                   'sequence_length': self.config.sequence_length,
                   'n_features': self.config.n_features,
                   'rul_ceiling': self.config.rul_ceiling,
                   'lstm_units': 64,
                   'dropout_rate': 0.2
               })
               
               # Prepare data
               print("Loading and preprocessing data...")
               X, y = self.preprocessor.prepare_training_data()
               
               # Split data
               X_train, X_test, y_train, y_test = train_test_split(
                   X, y, test_size=0.2, random_state=42
               )
               
               X_train, X_val, y_train, y_val = train_test_split(
                   X_train, y_train, test_size=0.2, random_state=42
               )
               
               print(f"Training data shape: {X_train.shape}")
               print(f"Validation data shape: {X_val.shape}")
               print(f"Test data shape: {X_test.shape}")
               
               # Build and train model
               print("Building model...")
               self.model.build_model(
                   input_shape=(self.config.sequence_length, 
                               self.config.n_features)
               )
               
               print("Training model...")
               history = self.model.train(
                   X_train, y_train, 
                   X_val, y_val,
                   epochs=100,
                   batch_size=256
               )
               
               # Evaluate model
               print("Evaluating model...")
               metrics = self.evaluator.evaluate(
                   self.model, X_test, y_test
               )
               
               # Log metrics
               mlflow.log_metrics(metrics)
               
               # Save model
               print("Saving model...")
               self.model.save_model(self.config.model_path)
               self.preprocessor.save_preprocessor(self.config.model_path)
               
               # Log model to MLflow
               mlflow.tensorflow.log_model(
                   self.model.model,
                   "lstm_rul_model"
               )
               
               print(f"Training complete. Metrics: {metrics}")
               
               return metrics
   ```

4. **Hyperparameter Tuning**
   ```python
   # src/models/hyperparameter_tuning.py
   import optuna
   from optuna.integration import TFKerasPruningCallback
   
   class HyperparameterTuner:
       def __init__(self, config):
           self.config = config
           
       def objective(self, trial):
           """Optuna objective function for hyperparameter tuning"""
           
           # Suggest hyperparameters
           lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128)
           lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128)
           dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
           learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
           batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
           
           # Build model with suggested parameters
           model = self.build_model_with_params(
               lstm_units_1, lstm_units_2, 
               dropout_rate, learning_rate
           )
           
           # Train model
           history = model.fit(
               self.X_train, self.y_train,
               validation_data=(self.X_val, self.y_val),
               epochs=50,
               batch_size=batch_size,
               callbacks=[TFKerasPruningCallback(trial, 'val_loss')],
               verbose=0
           )
           
           # Return validation loss
           return min(history.history['val_loss'])
       
       def tune(self, n_trials=50):
           """Run hyperparameter tuning"""
           study = optuna.create_study(
               direction='minimize',
               pruner=optuna.pruners.MedianPruner()
           )
           
           study.optimize(self.objective, n_trials=n_trials)
           
           print(f"Best parameters: {study.best_params}")
           print(f"Best value: {study.best_value}")
           
           return study.best_params
   ```

### Deliverables
- LSTM model implementation
- Training pipeline
- Evaluation metrics
- Hyperparameter tuning

### Testing Checklist
- [ ] Model builds correctly
- [ ] Training completes
- [ ] Metrics calculated
- [ ] Model saves/loads

---

## Phase 4: gRPC Server Implementation
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Implement gRPC server
- Handle protocol buffer communication
- Integrate with model inference

### Tasks
1. **Protocol Buffer Compilation**
   ```bash
   # compile_proto.sh
   python -m grpc_tools.protoc \
       -I./proto \
       --python_out=./src/grpc_server \
       --grpc_python_out=./src/grpc_server \
       ./proto/prediction.proto
   ```

2. **gRPC Server Implementation**
   ```python
   # src/grpc_server/server.py
   import grpc
   from concurrent import futures
   import logging
   import numpy as np
   
   from . import prediction_pb2
   from . import prediction_pb2_grpc
   from ..inference.predictor import ModelPredictor
   
   class PredictionServicer(prediction_pb2_grpc.PredictionServiceServicer):
       def __init__(self, predictor: ModelPredictor):
           self.predictor = predictor
           self.logger = logging.getLogger(__name__)
           
       def PredictRUL(self, request, context):
           """Handle single RUL prediction request"""
           try:
               # Extract sensor data from request
               sensor_data = []
               for reading in request.sensor_data:
                   sensor_dict = {
                       's2': reading.s2,
                       's3': reading.s3,
                       's4': reading.s4,
                       's7': reading.s7,
                       's8': reading.s8,
                       's9': reading.s9,
                       's11': reading.s11,
                       's12': reading.s12,
                       's13': reading.s13,
                       's14': reading.s14,
                       's15': reading.s15,
                       's17': reading.s17,
                       's20': reading.s20,
                       's21': reading.s21,
                   }
                   sensor_data.append(sensor_dict)
               
               # Make prediction
               predicted_rul, confidence = self.predictor.predict(
                   engine_id=request.engine_id,
                   sensor_data=sensor_data
               )
               
               # Create response
               response = prediction_pb2.RULResponse(
                   predicted_rul=int(predicted_rul),
                   confidence=float(confidence),
                   model_version=self.predictor.model_version
               )
               
               self.logger.info(
                   f"Prediction for engine {request.engine_id}: "
                   f"RUL={predicted_rul}, Confidence={confidence}"
               )
               
               return response
               
           except Exception as e:
               self.logger.error(f"Error in PredictRUL: {str(e)}")
               context.set_code(grpc.StatusCode.INTERNAL)
               context.set_details(f"Prediction failed: {str(e)}")
               return prediction_pb2.RULResponse()
       
       def BatchPredictRUL(self, request, context):
           """Handle batch prediction requests"""
           responses = []
           
           for engine_request in request.requests:
               response = self.PredictRUL(engine_request, context)
               responses.append(response)
           
           return prediction_pb2.BatchRULResponse(responses=responses)
   
   class GRPCServer:
       def __init__(self, config, predictor):
           self.config = config
           self.predictor = predictor
           self.server = None
           
       def start(self):
           """Start the gRPC server"""
           self.server = grpc.server(
               futures.ThreadPoolExecutor(max_workers=10)
           )
           
           # Add servicer
           prediction_pb2_grpc.add_PredictionServiceServicer_to_server(
               PredictionServicer(self.predictor),
               self.server
           )
           
           # Bind to port
           self.server.add_insecure_port(f'[::]:{self.config.grpc_port}')
           
           # Start server
           self.server.start()
           logging.info(f"gRPC server started on port {self.config.grpc_port}")
           
       def stop(self):
           """Stop the gRPC server"""
           if self.server:
               self.server.stop(grace_period=5)
               logging.info("gRPC server stopped")
   ```

3. **gRPC Client (for testing)**
   ```python
   # src/grpc_server/client.py
   import grpc
   from . import prediction_pb2
   from . import prediction_pb2_grpc
   
   class PredictionClient:
       def __init__(self, host='localhost', port=50051):
           self.channel = grpc.insecure_channel(f'{host}:{port}')
           self.stub = prediction_pb2_grpc.PredictionServiceStub(self.channel)
       
       def predict_rul(self, engine_id, sensor_data):
           """Make RUL prediction via gRPC"""
           
           # Create sensor readings
           sensor_readings = []
           for data in sensor_data:
               reading = prediction_pb2.SensorReading(**data)
               sensor_readings.append(reading)
           
           # Create request
           request = prediction_pb2.RULRequest(
               engine_id=engine_id,
               sensor_data=sensor_readings
           )
           
           # Make gRPC call
           response = self.stub.PredictRUL(request)
           
           return {
               'predicted_rul': response.predicted_rul,
               'confidence': response.confidence,
               'model_version': response.model_version
           }
       
       def close(self):
           """Close the gRPC channel"""
           self.channel.close()
   ```

### Deliverables
- gRPC server implementation
- Protocol buffer handling
- Client for testing
- Server management

### Testing Checklist
- [ ] Proto files compile
- [ ] Server starts correctly
- [ ] Client can connect
- [ ] Predictions work via gRPC

---

## Phase 5: Model Inference Service
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Implement inference pipeline
- Integrate with SageMaker
- Add caching and optimization

### Tasks
1. **Model Predictor**
   ```python
   # src/inference/predictor.py
   import numpy as np
   import tensorflow as tf
   from typing import List, Tuple, Optional
   import joblib
   import time
   from functools import lru_cache
   
   class ModelPredictor:
       def __init__(self, config):
           self.config = config
           self.model = None
           self.preprocessor = None
           self.model_version = "1.0.0"
           self.cache = {}
           self.load_model()
           
       def load_model(self):
           """Load model and preprocessor"""
           try:
               # Load Keras model
               self.model = tf.keras.models.load_model(
                   f"{self.config.model_path}/lstm_rul_model.h5"
               )
               
               # Load preprocessor
               self.preprocessor = joblib.load(
                   f"{self.config.model_path}/feature_engineer.pkl"
               )
               
               logging.info("Model and preprocessor loaded successfully")
               
           except Exception as e:
               logging.error(f"Failed to load model: {e}")
               raise
       
       def predict(self, engine_id: str, 
                  sensor_data: List[dict]) -> Tuple[float, float]:
           """Make RUL prediction for an engine"""
           
           # Check cache
           cache_key = self._get_cache_key(engine_id, sensor_data)
           if cache_key in self.cache:
               cached = self.cache[cache_key]
               if time.time() - cached['timestamp'] < 300:  # 5 min cache
                   return cached['prediction'], cached['confidence']
           
           # Preprocess data
           X = self.preprocessor.preprocess_for_inference(sensor_data)
           
           # Make prediction
           start_time = time.time()
           prediction = self.model.predict(X, verbose=0)
           inference_time = time.time() - start_time
           
           # Calculate confidence (based on prediction variance)
           confidence = self._calculate_confidence(prediction)
           
           # Extract scalar value
           rul_value = float(prediction[0][0])
           
           # Ensure non-negative
           rul_value = max(0, rul_value)
           
           # Cache result
           self.cache[cache_key] = {
               'prediction': rul_value,
               'confidence': confidence,
               'timestamp': time.time()
           }
           
           logging.info(
               f"Prediction for {engine_id}: RUL={rul_value:.2f}, "
               f"Confidence={confidence:.2f}, Time={inference_time:.3f}s"
           )
           
           return rul_value, confidence
       
       def _calculate_confidence(self, prediction: np.ndarray) -> float:
           """Calculate prediction confidence"""
           # Simple confidence based on prediction value
           # Lower RUL predictions are generally more confident
           rul = prediction[0][0]
           
           if rul < 20:
               confidence = 0.95
           elif rul < 50:
               confidence = 0.85
           elif rul < 100:
               confidence = 0.75
           else:
               confidence = 0.65
           
           return confidence
       
       def _get_cache_key(self, engine_id: str, 
                         sensor_data: List[dict]) -> str:
           """Generate cache key from input data"""
           # Use last reading as part of key
           last_reading = sensor_data[-1] if sensor_data else {}
           key_parts = [engine_id] + [str(v) for v in last_reading.values()]
           return '_'.join(key_parts)
       
       def warm_up(self):
           """Warm up the model with a dummy prediction"""
           dummy_data = [
               {f's{i}': np.random.rand() for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]}
               for _ in range(50)
           ]
           
           self.predict("warm_up", dummy_data)
           logging.info("Model warmed up")
   ```

2. **SageMaker Integration**
   ```python
   # src/sagemaker/sagemaker_client.py
   import boto3
   import json
   import numpy as np
   from typing import Optional
   
   class SageMakerPredictor:
       def __init__(self, config):
           self.config = config
           self.runtime_client = boto3.client(
               'sagemaker-runtime',
               region_name=config.aws_region
           )
           self.endpoint_name = config.sagemaker_endpoint
           
       def predict(self, input_data: np.ndarray) -> float:
           """Make prediction using SageMaker endpoint"""
           
           if not self.endpoint_name:
               raise ValueError("SageMaker endpoint not configured")
           
           try:
               # Prepare payload
               payload = json.dumps({
                   'instances': input_data.tolist()
               })
               
               # Invoke endpoint
               response = self.runtime_client.invoke_endpoint(
                   EndpointName=self.endpoint_name,
                   ContentType='application/json',
                   Body=payload
               )
               
               # Parse response
               result = json.loads(response['Body'].read().decode())
               prediction = result['predictions'][0][0]
               
               return float(prediction)
               
           except Exception as e:
               logging.error(f"SageMaker prediction failed: {e}")
               raise
       
       def get_endpoint_config(self):
           """Get endpoint configuration"""
           sagemaker_client = boto3.client(
               'sagemaker',
               region_name=self.config.aws_region
           )
           
           response = sagemaker_client.describe_endpoint(
               EndpointName=self.endpoint_name
           )
           
           return {
               'status': response['EndpointStatus'],
               'model_name': response['EndpointConfigName'],
               'created_time': response['CreationTime'],
               'last_modified': response['LastModifiedTime']
           }
   ```

3. **Hybrid Predictor (Local + SageMaker)**
   ```python
   # src/inference/hybrid_predictor.py
   class HybridPredictor:
       def __init__(self, config):
           self.config = config
           self.local_predictor = ModelPredictor(config)
           self.sagemaker_predictor = None
           
           if config.sagemaker_endpoint:
               self.sagemaker_predictor = SageMakerPredictor(config)
       
       def predict(self, engine_id: str, 
                  sensor_data: List[dict],
                  use_sagemaker: bool = False) -> Tuple[float, float]:
           """Make prediction using local or SageMaker model"""
           
           if use_sagemaker and self.sagemaker_predictor:
               # Use SageMaker endpoint
               X = self.local_predictor.preprocessor.preprocess_for_inference(
                   sensor_data
               )
               prediction = self.sagemaker_predictor.predict(X)
               confidence = self.local_predictor._calculate_confidence(
                   np.array([[prediction]])
               )
               return prediction, confidence
           else:
               # Use local model
               return self.local_predictor.predict(engine_id, sensor_data)
   ```

### Deliverables
- Model inference service
- SageMaker integration
- Caching mechanism
- Hybrid predictor

### Testing Checklist
- [ ] Local predictions work
- [ ] SageMaker integration works
- [ ] Caching improves performance
- [ ] Error handling robust

---

## Phase 6: FastAPI REST Interface
**Duration**: 3-4 days  
**Priority**: Medium

### Objectives
- Create REST API endpoints
- Add health checks
- Implement monitoring

### Tasks
1. **FastAPI Application**
   ```python
   # src/api/app.py
   from fastapi import FastAPI, HTTPException, status
   from fastapi.middleware.cors import CORSMiddleware
   from contextlib import asynccontextmanager
   import structlog
   
   from ..config.settings import settings
   from ..inference.hybrid_predictor import HybridPredictor
   from .models import PredictionRequest, PredictionResponse
   
   logger = structlog.get_logger()
   
   # Global predictor instance
   predictor = None
   
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Startup
       global predictor
       logger.info("Starting ML service...")
       predictor = HybridPredictor(settings)
       predictor.local_predictor.warm_up()
       yield
       # Shutdown
       logger.info("Shutting down ML service...")
   
   app = FastAPI(
       title="RUL Prediction Service",
       description="ML service for predicting Remaining Useful Life",
       version="1.0.0",
       lifespan=lifespan
   )
   
   # CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   @app.get("/health")
   async def health_check():
       """Health check endpoint"""
       return {
           "status": "healthy",
           "service": settings.service_name,
           "version": "1.0.0"
       }
   
   @app.get("/ready")
   async def readiness_check():
       """Readiness check endpoint"""
       if predictor and predictor.local_predictor.model:
           return {"ready": True}
       raise HTTPException(
           status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
           detail="Service not ready"
       )
   
   @app.post("/predict", response_model=PredictionResponse)
   async def predict_rul(request: PredictionRequest):
       """Predict RUL for an engine"""
       try:
           # Convert to dict format
           sensor_data = [reading.dict() for reading in request.sensor_data]
           
           # Make prediction
           rul, confidence = predictor.predict(
               engine_id=request.engine_id,
               sensor_data=sensor_data,
               use_sagemaker=request.use_sagemaker
           )
           
           return PredictionResponse(
               engine_id=request.engine_id,
               predicted_rul=rul,
               confidence=confidence,
               model_version=predictor.local_predictor.model_version
           )
           
       except Exception as e:
           logger.error(f"Prediction failed: {e}")
           raise HTTPException(
               status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
               detail=f"Prediction failed: {str(e)}"
           )
   
   @app.get("/metrics")
   async def get_metrics():
       """Get service metrics"""
       return {
           "cache_size": len(predictor.local_predictor.cache),
           "model_version": predictor.local_predictor.model_version,
           "total_predictions": 0,  # Implement counter
           "avg_inference_time": 0  # Implement timing
       }
   ```

2. **Request/Response Models**
   ```python
   # src/api/models.py
   from pydantic import BaseModel, Field
   from typing import List, Optional
   from datetime import datetime
   
   class SensorReading(BaseModel):
       s2: float = Field(..., description="T24 - Total temp at LPC outlet")
       s3: float = Field(..., description="T30 - Total temp at HPC outlet")
       s4: float = Field(..., description="T50 - Total temp at LPT outlet")
       s7: float = Field(..., description="P30 - Total pressure at HPC outlet")
       s8: float = Field(..., description="Nf - Physical fan speed")
       s9: float = Field(..., description="Nc - Physical core speed")
       s11: float = Field(..., description="Ps30 - Static pressure at HPC outlet")
       s12: float = Field(..., description="phi - Fuel flow ratio")
       s13: float = Field(..., description="NRf - Corrected fan speed")
       s14: float = Field(..., description="NRc - Corrected core speed")
       s15: float = Field(..., description="BPR - Bypass Ratio")
       s17: float = Field(..., description="htBleed - HPT coolant bleed")
       s20: float = Field(..., description="W31 - LPT coolant bleed")
       s21: float = Field(..., description="W32 - HPT coolant bleed")
   
   class PredictionRequest(BaseModel):
       engine_id: str = Field(..., description="Engine identifier")
       sensor_data: List[SensorReading] = Field(
           ..., 
           description="Sequence of sensor readings"
       )
       use_sagemaker: bool = Field(
           default=False,
           description="Use SageMaker endpoint instead of local model"
       )
   
   class PredictionResponse(BaseModel):
       engine_id: str
       predicted_rul: float = Field(..., description="Predicted RUL in cycles")
       confidence: float = Field(..., description="Prediction confidence (0-1)")
       model_version: str
       timestamp: datetime = Field(default_factory=datetime.utcnow)
   ```

### Deliverables
- FastAPI application
- REST endpoints
- Health checks
- Metrics endpoint

### Testing Checklist
- [ ] FastAPI starts correctly
- [ ] Endpoints respond
- [ ] Health checks work
- [ ] Error handling works

---

## Phase 7: Testing and Quality Assurance
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Implement comprehensive testing
- Performance testing
- Integration testing

### Tasks
1. **Unit Tests**
   ```python
   # tests/unit/test_preprocessing.py
   import pytest
   import numpy as np
   from src.preprocessing.feature_engineering import FeatureEngineer
   
   def test_feature_engineer_initialization():
       fe = FeatureEngineer(sequence_length=50)
       assert fe.sequence_length == 50
       assert fe.scaler is not None
   
   def test_create_sequences():
       fe = FeatureEngineer(sequence_length=3)
       data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
       labels = np.array([10, 20, 30, 40])
       
       sequences, targets = fe.create_sequences(data, labels)
       
       assert sequences.shape == (2, 3, 2)
       assert targets.shape == (2,)
       assert np.array_equal(targets, [30, 40])
   ```

2. **Integration Tests**
   ```python
   # tests/integration/test_grpc_server.py
   import pytest
   import grpc
   from src.grpc_server.server import GRPCServer
   from src.grpc_server.client import PredictionClient
   
   @pytest.fixture
   def grpc_server():
       # Setup
       server = GRPCServer(config, predictor)
       server.start()
       yield server
       # Teardown
       server.stop()
   
   def test_grpc_prediction(grpc_server):
       client = PredictionClient()
       
       sensor_data = [
           {'s2': 642.1, 's3': 1580.2, ...}
           for _ in range(50)
       ]
       
       result = client.predict_rul('engine-1', sensor_data)
       
       assert 'predicted_rul' in result
       assert result['predicted_rul'] >= 0
       assert 0 <= result['confidence'] <= 1
   ```

3. **Performance Tests**
   ```python
   # tests/performance/test_inference_speed.py
   import time
   import statistics
   
   def test_inference_latency():
       predictor = ModelPredictor(config)
       
       # Prepare test data
       sensor_data = generate_test_data()
       
       # Warm up
       predictor.predict('warm-up', sensor_data)
       
       # Measure latencies
       latencies = []
       for i in range(100):
           start = time.time()
           predictor.predict(f'engine-{i}', sensor_data)
           latencies.append((time.time() - start) * 1000)
       
       # Assert performance requirements
       assert statistics.mean(latencies) < 100  # < 100ms average
       assert statistics.quantiles(latencies, n=100)[94] < 200  # P95 < 200ms
   ```

### Deliverables
- Unit test suite
- Integration tests
- Performance tests
- Test coverage report

### Testing Checklist
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance meets requirements
- [ ] Coverage > 80%

---

## Phase 8: Deployment and Operations
**Duration**: 4-5 days  
**Priority**: Critical

### Objectives
- Containerize service
- Deploy to AWS
- Set up monitoring

### Tasks
1. **Docker Configuration**
   - Refer to **Infrastructure as Code Implementation Plan** for complete Docker setup
   - Container configuration specific to ML service requirements
   - Multi-stage builds for optimized images

2. **CI/CD Pipeline**
   - See **Infrastructure as Code Implementation Plan** for complete CI/CD setup
   - ML service specific testing and deployment steps
   - Model artifact handling in deployment pipeline

3. **Monitoring Setup**
   - Refer to **Monitoring & Observability Implementation Plan** for comprehensive monitoring
   - ML-specific metrics collection (prediction latency, model performance)
   - Integration with centralized logging and alerting

4. **AWS Deployment**
   - See **Infrastructure as Code Implementation Plan** for AWS infrastructure
   - ECS Fargate deployment configuration
   - Auto-scaling policies for ML workloads

### Deliverables
- Docker configuration (see Infrastructure plan for details)
- CI/CD pipeline integration (see Infrastructure plan)
- Deployed service on AWS (see Infrastructure plan)
- Monitoring setup (see Monitoring plan for details)

### Testing Checklist
- [ ] Docker image builds (Infrastructure plan)
- [ ] Service starts in container
- [ ] CI/CD pipeline works (Infrastructure plan)
- [ ] Monitoring collects metrics (Monitoring plan)

---

## Success Metrics

### Performance KPIs
- Model inference latency: < 100ms
- gRPC response time: < 150ms
- Service availability: > 99.9%
- Memory usage: < 2GB

### Model Quality
- RMSE: < 20 cycles
- PHM08 Score: < 500
- Prediction consistency: > 95%

---

## Risk Mitigation

### Technical Risks
1. **Model Degradation**
   - Mitigation: Implement model monitoring and retraining pipeline

2. **High Latency**
   - Mitigation: Use model quantization and caching

3. **Memory Issues**
   - Mitigation: Implement batch processing and memory management

### Dependencies
1. **TensorFlow Version Conflicts**
   - Mitigation: Use Docker for consistent environment

2. **SageMaker Availability**
   - Mitigation: Fallback to local model

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Setup | 2-3 days | None |
| Phase 2: Data Processing | 4-5 days | Phase 1 |
| Phase 3: Model Development | 5-6 days | Phase 2 |
| Phase 4: gRPC Server | 3-4 days | Phase 1 |
| Phase 5: Inference Service | 4-5 days | Phase 3, 4 |
| Phase 6: FastAPI | 3-4 days | Phase 5 |
| Phase 7: Testing | 4-5 days | All phases |
| Phase 8: Deployment | 4-5 days | Phase 7 |

**Total Duration**: 29-37 days (6-7 weeks)

---

## Next Steps
1. Review and approve implementation plan
2. Set up Python development environment
3. Begin Phase 1 implementation
4. Prepare C-MAPSS dataset
5. Coordinate with API Gateway team for gRPC interface