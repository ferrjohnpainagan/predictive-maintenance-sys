# Python ML Service Free Tier Implementation Plan - Predictive Maintenance System

## Project Overview

Implementation plan for the Python ML microservice using AWS Free Tier resources. This approach focuses on EC2-based deployment and local model serving instead of AWS SageMaker to stay within free tier limits while maintaining full ML functionality.

## Technology Stack (Free Tier)

- **Language**: Python 3.9+
- **Web Framework**: FastAPI
- **ML Framework**: TensorFlow/Keras (CPU optimized)
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Communication**: gRPC (grpcio)
- **Model Serving**: Local TensorFlow Serving
- **Testing**: pytest, pytest-asyncio
- **Deployment**: EC2 t2.micro (Free tier eligible)

## Free Tier Considerations

- **EC2**: Using t2.micro instances (1 vCPU, 1GB RAM)
- **Storage**: S3 Free tier (5GB storage)
- **ML Models**: Lightweight models optimized for CPU inference
- **Cost**: $0/month for 12 months

---

## Phase 1: Project Setup and Foundation

**Duration**: 2-3 days  
**Priority**: Critical

### Objectives

- Initialize Python project optimized for free tier
- Set up development environment for CPU-only inference
- Configure dependencies for minimal resource usage

### Tasks

1. **Project Initialization**

   ```bash
   # Create project structure
   mkdir backend/ml-service
   cd backend/ml-service

   # Initialize Python project
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Create project structure
   mkdir -p {src/{models,grpc,api,utils,config},tests,deployment,data,notebooks}
   ```

2. **Dependencies Installation (Free Tier Optimized)**

   ```toml
   # pyproject.toml
   [tool.poetry.dependencies]
   python = "^3.9"

   # Core ML dependencies (CPU optimized)
   tensorflow-cpu = "^2.14.0"  # CPU-only version for free tier
   numpy = "^1.24.0"
   pandas = "^2.1.0"
   scikit-learn = "^1.3.0"

   # Web framework
   fastapi = "^0.104.0"
   uvicorn = {extras = ["standard"], version = "^0.24.0"}

   # gRPC
   grpcio = "^1.59.0"
   grpcio-tools = "^1.59.0"

   # AWS (free tier)
   boto3 = "^1.28.0"

   # Utilities
   pydantic = "^2.4.0"
   python-dotenv = "^1.0.0"
   structlog = "^23.1.0"

   # Free tier optimizations
   psutil = "^5.9.0"  # System monitoring

   [tool.poetry.group.dev.dependencies]
   pytest = "^7.4.0"
   pytest-asyncio = "^0.21.0"
   black = "^23.9.0"
   flake8 = "^6.0.0"
   mypy = "^1.5.0"
   ```

3. **Project Structure (Free Tier Focused)**

   ```
   backend/ml-service/
   ├── src/
   │   ├── models/
   │   │   ├── __init__.py
   │   │   ├── lstm_model.py          # Lightweight LSTM
   │   │   ├── model_loader.py        # Local model loading
   │   │   └── inference_engine.py    # CPU-optimized inference
   │   ├── grpc/
   │   │   ├── __init__.py
   │   │   ├── server.py              # gRPC server
   │   │   ├── servicer.py            # ML service implementation
   │   │   └── proto/                 # Protocol buffers
   │   ├── api/
   │   │   ├── __init__.py
   │   │   ├── health.py              # Health endpoints
   │   │   └── prediction.py          # REST API endpoints
   │   ├── utils/
   │   │   ├── __init__.py
   │   │   ├── preprocessing.py       # Data preprocessing
   │   │   ├── monitoring.py          # Resource monitoring
   │   │   └── cache.py               # Simple caching
   │   ├── config/
   │   │   ├── __init__.py
   │   │   └── settings.py            # Configuration
   │   └── main.py                    # Application entry point
   ├── deployment/
   │   ├── docker/
   │   │   └── Dockerfile.free-tier   # Optimized for t2.micro
   │   ├── scripts/
   │   │   ├── deploy-ec2.sh          # EC2 deployment
   │   │   └── train-local.py         # Local model training
   │   └── systemd/
   │       └── ml-service.service     # Service configuration
   ├── data/
   │   ├── raw/                       # Raw training data
   │   ├── processed/                 # Preprocessed data
   │   └── models/                    # Trained models
   ├── tests/
   ├── notebooks/                     # Jupyter notebooks for development
   └── requirements.txt
   ```

4. **Configuration (Free Tier)**

   ```python
   # src/config/settings.py
   from pydantic import BaseSettings
   import os

   class Settings(BaseSettings):
       # Server configuration
       host: str = "0.0.0.0"
       port: int = 50051
       grpc_port: int = 50051
       http_port: int = 8000

       # Model configuration (free tier optimized)
       model_path: str = "/app/models"
       model_name: str = "lstm_rul_model"
       batch_size: int = 1  # Single prediction for t2.micro
       max_sequence_length: int = 50

       # Performance settings (t2.micro limits)
       max_workers: int = 2  # Limited for 1GB RAM
       memory_limit_mb: int = 700  # Leave 300MB for system

       # Caching (in-memory for free tier)
       cache_enabled: bool = True
       cache_ttl_seconds: int = 300
       max_cache_entries: int = 100

       # AWS configuration (free tier)
       aws_region: str = "us-east-1"
       s3_bucket: str = os.getenv("S3_BUCKET", "")

       # Monitoring
       enable_metrics: bool = True
       log_level: str = "INFO"

       class Config:
           env_file = ".env"

   settings = Settings()
   ```

### Deliverables

- [ ] Python project initialized
- [ ] Dependencies configured for free tier
- [ ] Project structure established
- [ ] Configuration optimized for t2.micro

---

## Phase 2: Data Processing Pipeline (Free Tier)

**Duration**: 4-5 days  
**Priority**: High

### Objectives

- Implement lightweight data preprocessing
- Create efficient data loading for small memory footprint
- Optimize for CPU-only operations

### Tasks

1. **Data Preprocessing (Memory Optimized)**

   ```python
   # src/utils/preprocessing.py
   import numpy as np
   import pandas as pd
   from typing import Tuple, List
   import logging

   logger = logging.getLogger(__name__)

   class DataPreprocessor:
       def __init__(self, sequence_length: int = 50):
           self.sequence_length = sequence_length
           self.scaler = None

       def preprocess_sensor_data(self, data: List[dict]) -> np.ndarray:
           """
           Preprocess sensor data for RUL prediction.
           Optimized for low memory usage on t2.micro.
           """
           try:
               # Convert to DataFrame with memory optimization
               df = pd.DataFrame(data, dtype=np.float32)  # Use float32 to save memory

               # Basic feature engineering (lightweight)
               df['sensor_mean'] = df.iloc[:, 1:].mean(axis=1)
               df['sensor_std'] = df.iloc[:, 1:].std(axis=1)

               # Normalization (fit scaler on first use)
               if self.scaler is None:
                   from sklearn.preprocessing import MinMaxScaler
                   self.scaler = MinMaxScaler()
                   normalized_data = self.scaler.fit_transform(df)
               else:
                   normalized_data = self.scaler.transform(df)

               # Create sequences for LSTM (memory efficient)
               sequences = self._create_sequences(normalized_data)

               return sequences

           except Exception as e:
               logger.error(f"Preprocessing error: {e}")
               raise

       def _create_sequences(self, data: np.ndarray) -> np.ndarray:
           """Create sequences for LSTM input with minimal memory usage."""
           if len(data) < self.sequence_length:
               # Pad if insufficient data
               padded = np.zeros((self.sequence_length, data.shape[1]), dtype=np.float32)
               padded[:len(data)] = data
               return padded.reshape(1, self.sequence_length, -1)

           # Take last sequence_length points
           sequence = data[-self.sequence_length:].astype(np.float32)
           return sequence.reshape(1, self.sequence_length, -1)
   ```

2. **Model Definition (Lightweight LSTM)**

   ```python
   # src/models/lstm_model.py
   import tensorflow as tf
   import numpy as np
   from typing import Optional
   import logging

   logger = logging.getLogger(__name__)

   class LightweightLSTMModel:
       """Lightweight LSTM model optimized for CPU inference on t2.micro."""

       def __init__(self, sequence_length: int = 50, num_features: int = 14):
           self.sequence_length = sequence_length
           self.num_features = num_features
           self.model: Optional[tf.keras.Model] = None

       def create_model(self) -> tf.keras.Model:
           """Create a lightweight LSTM model for RUL prediction."""

           # Optimized architecture for t2.micro
           model = tf.keras.Sequential([
               tf.keras.layers.LSTM(
                   32,  # Reduced units for memory efficiency
                   input_shape=(self.sequence_length, self.num_features),
                   return_sequences=False,
                   dropout=0.2
               ),
               tf.keras.layers.Dense(16, activation='relu'),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(1, activation='linear')  # RUL output
           ])

           # Use lightweight optimizer
           model.compile(
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss='mse',
               metrics=['mae']
           )

           self.model = model
           return model

       def predict(self, input_data: np.ndarray) -> float:
           """Make prediction with error handling for t2.micro constraints."""
           try:
               # Ensure input is float32 for memory efficiency
               input_data = input_data.astype(np.float32)

               # Make prediction
               prediction = self.model.predict(input_data, verbose=0)

               # Return scalar value
               return float(prediction[0][0])

           except Exception as e:
               logger.error(f"Prediction error: {e}")
               raise
   ```

3. **Model Loader (Local Storage)**

   ```python
   # src/models/model_loader.py
   import tensorflow as tf
   import os
   import pickle
   from typing import Tuple, Optional
   import logging
   from .lstm_model import LightweightLSTMModel

   logger = logging.getLogger(__name__)

   class ModelLoader:
       """Load and manage models locally for free tier deployment."""

       def __init__(self, model_path: str):
           self.model_path = model_path
           self.model: Optional[LightweightLSTMModel] = None
           self.scaler = None

       def load_model(self) -> LightweightLSTMModel:
           """Load pre-trained model from local storage."""
           try:
               model_file = os.path.join(self.model_path, "lstm_model.h5")
               scaler_file = os.path.join(self.model_path, "scaler.pkl")

               if not os.path.exists(model_file):
                   # Create and train a basic model if none exists
                   logger.warning("No pre-trained model found. Creating basic model.")
                   return self._create_basic_model()

               # Load TensorFlow model
               tf_model = tf.keras.models.load_model(model_file)

               # Load scaler
               if os.path.exists(scaler_file):
                   with open(scaler_file, 'rb') as f:
                       self.scaler = pickle.load(f)

               # Wrap in our model class
               self.model = LightweightLSTMModel()
               self.model.model = tf_model

               logger.info("Model loaded successfully")
               return self.model

           except Exception as e:
               logger.error(f"Model loading failed: {e}")
               return self._create_basic_model()

       def _create_basic_model(self) -> LightweightLSTMModel:
           """Create a basic model for development/testing."""
           logger.info("Creating basic model for development")

           model = LightweightLSTMModel()
           model.create_model()

           # Save for future use
           self._save_model(model)

           return model

       def _save_model(self, model: LightweightLSTMModel):
           """Save model to local storage."""
           try:
               os.makedirs(self.model_path, exist_ok=True)

               model_file = os.path.join(self.model_path, "lstm_model.h5")
               model.model.save(model_file)

               logger.info("Model saved successfully")

           except Exception as e:
               logger.error(f"Model saving failed: {e}")
   ```

### Deliverables

- [ ] Data preprocessing pipeline implemented
- [ ] Lightweight LSTM model created
- [ ] Model loading system configured

---

## Phase 3: gRPC Server Implementation (Free Tier)

**Duration**: 3-4 days  
**Priority**: High

### Objectives

- Implement gRPC server optimized for t2.micro
- Create efficient communication with API Gateway
- Add resource monitoring and limiting

### Tasks

1. **gRPC Service Implementation**

   ```python
   # src/grpc/servicer.py
   import grpc
   from concurrent import futures
   import logging
   import psutil
   import asyncio
   from typing import Dict, Any

   from .proto import ml_service_pb2
   from .proto import ml_service_pb2_grpc
   from ..models.model_loader import ModelLoader
   from ..utils.preprocessing import DataPreprocessor
   from ..utils.monitoring import ResourceMonitor

   logger = logging.getLogger(__name__)

   class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
       """gRPC servicer optimized for t2.micro constraints."""

       def __init__(self, model_path: str):
           self.model_loader = ModelLoader(model_path)
           self.model = self.model_loader.load_model()
           self.preprocessor = DataPreprocessor()
           self.resource_monitor = ResourceMonitor()

           # Simple in-memory cache for t2.micro
           self.prediction_cache: Dict[str, Any] = {}
           self.cache_max_size = 100

       def PredictRUL(self, request, context):
           """Predict Remaining Useful Life with resource monitoring."""
           try:
               # Check system resources before processing
               if not self._check_resources():
                   context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                   context.set_details("System resources exhausted")
                   return ml_service_pb2.PredictionResponse()

               # Check cache first
               cache_key = self._generate_cache_key(request)
               if cache_key in self.prediction_cache:
                   logger.info("Returning cached prediction")
                   return self.prediction_cache[cache_key]

               # Preprocess sensor data
               sensor_data = []
               for sensor in request.sensors:
                   sensor_data.append({
                       'sensor_id': sensor.sensor_id,
                       'value': sensor.value,
                       'timestamp': sensor.timestamp
                   })

               # Prepare input for model
               processed_data = self.preprocessor.preprocess_sensor_data(sensor_data)

               # Make prediction
               predicted_rul = self.model.predict(processed_data)

               # Create response
               response = ml_service_pb2.PredictionResponse(
                   engine_id=request.engine_id,
                   predicted_rul=predicted_rul,
                   confidence=0.85,  # Mock confidence for now
                   model_version="1.0.0"
               )

               # Cache response
               self._cache_response(cache_key, response)

               return response

           except Exception as e:
               logger.error(f"Prediction failed: {e}")
               context.set_code(grpc.StatusCode.INTERNAL)
               context.set_details(f"Prediction failed: {str(e)}")
               return ml_service_pb2.PredictionResponse()

       def HealthCheck(self, request, context):
           """Health check with system resource information."""
           try:
               memory_percent = psutil.virtual_memory().percent
               cpu_percent = psutil.cpu_percent(interval=1)

               status = "SERVING"
               if memory_percent > 85 or cpu_percent > 90:
                   status = "NOT_SERVING"

               return ml_service_pb2.HealthStatus(
                   status=status,
                   memory_usage=memory_percent,
                   cpu_usage=cpu_percent
               )

           except Exception as e:
               logger.error(f"Health check failed: {e}")
               context.set_code(grpc.StatusCode.INTERNAL)
               return ml_service_pb2.HealthStatus(status="NOT_SERVING")

       def _check_resources(self) -> bool:
           """Check if system has enough resources for processing."""
           memory_percent = psutil.virtual_memory().percent
           return memory_percent < 80  # Leave 20% memory free

       def _generate_cache_key(self, request) -> str:
           """Generate cache key from request."""
           sensor_values = [f"{s.sensor_id}:{s.value}" for s in request.sensors]
           return f"{request.engine_id}:{'_'.join(sensor_values)}"

       def _cache_response(self, key: str, response):
           """Cache response with size limit."""
           if len(self.prediction_cache) >= self.cache_max_size:
               # Remove oldest entry
               oldest_key = next(iter(self.prediction_cache))
               del self.prediction_cache[oldest_key]

           self.prediction_cache[key] = response
   ```

2. **gRPC Server (Resource Limited)**

   ```python
   # src/grpc/server.py
   import grpc
   from concurrent import futures
   import logging
   import signal
   import sys
   from typing import Optional

   from .servicer import MLServiceServicer
   from .proto import ml_service_pb2_grpc
   from ..config.settings import settings

   logger = logging.getLogger(__name__)

   class GRPCServer:
       """gRPC server optimized for t2.micro constraints."""

       def __init__(self):
           self.server: Optional[grpc.Server] = None
           self.servicer: Optional[MLServiceServicer] = None

       def start_server(self):
           """Start gRPC server with resource limits."""
           try:
               # Create server with limited thread pool for t2.micro
               self.server = grpc.server(
                   futures.ThreadPoolExecutor(max_workers=settings.max_workers),
                   options=[
                       ('grpc.keepalive_time_ms', 30000),
                       ('grpc.keepalive_timeout_ms', 5000),
                       ('grpc.keepalive_permit_without_calls', True),
                       ('grpc.http2.max_pings_without_data', 0),
                       ('grpc.http2.min_time_between_pings_ms', 10000),
                       ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                       # Memory optimization for t2.micro
                       ('grpc.max_receive_message_length', 1024 * 1024),  # 1MB
                       ('grpc.max_send_message_length', 1024 * 1024),     # 1MB
                   ]
               )

               # Add servicer
               self.servicer = MLServiceServicer(settings.model_path)
               ml_service_pb2_grpc.add_MLServiceServicer_to_server(
                   self.servicer, self.server
               )

               # Listen on all interfaces
               listen_addr = f"{settings.host}:{settings.grpc_port}"
               self.server.add_insecure_port(listen_addr)

               # Start server
               self.server.start()
               logger.info(f"gRPC server started on {listen_addr}")

               # Setup signal handlers
               self._setup_signal_handlers()

               # Wait for termination
               self.server.wait_for_termination()

           except Exception as e:
               logger.error(f"Failed to start gRPC server: {e}")
               raise

       def stop_server(self):
           """Gracefully stop the server."""
           if self.server:
               logger.info("Stopping gRPC server...")
               self.server.stop(grace=10)

       def _setup_signal_handlers(self):
           """Setup signal handlers for graceful shutdown."""
           def signal_handler(signum, frame):
               logger.info(f"Received signal {signum}")
               self.stop_server()
               sys.exit(0)

           signal.signal(signal.SIGINT, signal_handler)
           signal.signal(signal.SIGTERM, signal_handler)
   ```

### Deliverables

- [ ] gRPC servicer implemented
- [ ] Server optimized for t2.micro
- [ ] Resource monitoring integrated

---

## Phase 4: FastAPI REST Interface (Free Tier)

**Duration**: 4-5 days  
**Priority**: Medium

### Objectives

- Create REST API for direct access
- Implement health monitoring endpoints
- Add resource usage monitoring

### Tasks

1. **FastAPI Application**

   ```python
   # src/api/main.py
   from fastapi import FastAPI, HTTPException, BackgroundTasks
   from fastapi.middleware.cors import CORSMiddleware
   import uvicorn
   import logging
   from typing import List, Dict, Any

   from .prediction import PredictionAPI
   from .health import HealthAPI
   from ..config.settings import settings

   logger = logging.getLogger(__name__)

   # Create FastAPI app
   app = FastAPI(
       title="Predictive Maintenance ML Service",
       description="Free Tier ML Service for RUL Prediction",
       version="1.0.0",
       docs_url="/docs" if settings.log_level == "DEBUG" else None,
   )

   # Add CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Configure appropriately for production
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

   # Initialize APIs
   prediction_api = PredictionAPI()
   health_api = HealthAPI()

   # Include routers
   app.include_router(prediction_api.router, prefix="/api/v1", tags=["predictions"])
   app.include_router(health_api.router, prefix="/health", tags=["health"])

   @app.get("/")
   async def root():
       """Root endpoint."""
       return {
           "service": "ML Service",
           "version": "1.0.0",
           "status": "running",
           "tier": "free"
       }

   if __name__ == "__main__":
       uvicorn.run(
           "main:app",
           host=settings.host,
           port=settings.http_port,
           log_level=settings.log_level.lower(),
           workers=1,  # Single worker for t2.micro
       )
   ```

2. **Health Monitoring API**

   ```python
   # src/api/health.py
   from fastapi import APIRouter, HTTPException
   import psutil
   import logging
   from typing import Dict, Any

   logger = logging.getLogger(__name__)

   class HealthAPI:
       def __init__(self):
           self.router = APIRouter()
           self._setup_routes()

       def _setup_routes(self):
           @self.router.get("/")
           async def health_check() -> Dict[str, Any]:
               """Basic health check."""
               try:
                   return {
                       "status": "healthy",
                       "service": "ml-service",
                       "tier": "free"
                   }
               except Exception as e:
                   logger.error(f"Health check failed: {e}")
                   raise HTTPException(status_code=503, detail="Service unhealthy")

           @self.router.get("/detailed")
           async def detailed_health() -> Dict[str, Any]:
               """Detailed health check with system metrics."""
               try:
                   # System metrics
                   memory = psutil.virtual_memory()
                   cpu_percent = psutil.cpu_percent(interval=1)
                   disk = psutil.disk_usage('/')

                   # Determine status
                   status = "healthy"
                   if memory.percent > 85 or cpu_percent > 90:
                       status = "degraded"
                   if memory.percent > 95 or cpu_percent > 95:
                       status = "unhealthy"

                   return {
                       "status": status,
                       "timestamp": int(psutil.boot_time()),
                       "system": {
                           "memory": {
                               "total": memory.total,
                               "available": memory.available,
                               "percent": memory.percent,
                               "used": memory.used
                           },
                           "cpu": {
                               "percent": cpu_percent,
                               "count": psutil.cpu_count()
                           },
                           "disk": {
                               "total": disk.total,
                               "used": disk.used,
                               "free": disk.free,
                               "percent": (disk.used / disk.total) * 100
                           }
                       },
                       "service": {
                           "tier": "free",
                           "model_loaded": True,  # Check if model is loaded
                           "grpc_port": 50051,
                           "http_port": 8000
                       }
                   }

               except Exception as e:
                   logger.error(f"Detailed health check failed: {e}")
                   raise HTTPException(status_code=503, detail="Service unhealthy")
   ```

### Deliverables

- [ ] FastAPI application created
- [ ] Health monitoring endpoints implemented
- [ ] Resource usage monitoring configured

---

## Phase 5: Free Tier Deployment Configuration

**Duration**: 4-5 days  
**Priority**: Critical

### Objectives

- Configure Docker for EC2 deployment
- Create deployment scripts optimized for t2.micro
- Set up service management

### Tasks

1. **Docker Configuration (Free Tier Optimized)**

   ```dockerfile
   # deployment/docker/Dockerfile.free-tier
   FROM python:3.9-slim as base

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # Set work directory
   WORKDIR /app

   # Copy requirements first for better caching
   COPY requirements.txt .

   # Install Python dependencies with optimization for t2.micro
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY data/models/ ./models/

   # Create non-root user
   RUN useradd --create-home --shell /bin/bash mlservice
   RUN chown -R mlservice:mlservice /app
   USER mlservice

   # Expose ports
   EXPOSE 50051 8000

   # Health check optimized for t2.micro
   HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
     CMD curl -f http://localhost:8000/health || exit 1

   # Set environment variables for production
   ENV PYTHONPATH=/app
   ENV PYTHONUNBUFFERED=1
   ENV TF_CPP_MIN_LOG_LEVEL=2

   # Start both gRPC and HTTP servers
   CMD ["python", "-m", "src.main"]
   ```

2. **Main Application Entry Point**

   ```python
   # src/main.py
   import asyncio
   import logging
   import multiprocessing
   from concurrent.futures import ThreadPoolExecutor

   from .grpc.server import GRPCServer
   from .api.main import app
   from .config.settings import settings
   import uvicorn

   logging.basicConfig(level=getattr(logging, settings.log_level))
   logger = logging.getLogger(__name__)

   async def run_grpc_server():
       """Run gRPC server in background."""
       grpc_server = GRPCServer()
       grpc_server.start_server()

   async def run_http_server():
       """Run HTTP server."""
       config = uvicorn.Config(
           app,
           host=settings.host,
           port=settings.http_port,
           log_level=settings.log_level.lower(),
           workers=1,  # Single worker for t2.micro
       )
       server = uvicorn.Server(config)
       await server.serve()

   def main():
       """Main entry point for both servers."""
       logger.info("Starting ML Service (Free Tier)")

       # Start gRPC server in separate process for t2.micro
       grpc_process = multiprocessing.Process(target=lambda: asyncio.run(run_grpc_server()))
       grpc_process.start()

       try:
           # Start HTTP server in main process
           asyncio.run(run_http_server())
       except KeyboardInterrupt:
           logger.info("Shutting down...")
       finally:
           grpc_process.terminate()
           grpc_process.join()

   if __name__ == "__main__":
       main()
   ```

3. **EC2 Deployment Script**

   ```bash
   #!/bin/bash
   # deployment/scripts/deploy-ec2.sh

   set -e

   echo "🚀 Deploying ML Service to EC2 (Free Tier)"

   # Variables
   APP_NAME="predictive-maintenance-ml"
   GRPC_PORT=50051
   HTTP_PORT=8000
   DOCKER_IMAGE="$APP_NAME:latest"

   # Build optimized image for t2.micro
   echo "📦 Building Docker image..."
   docker build -f deployment/docker/Dockerfile.free-tier -t $DOCKER_IMAGE .

   # Stop existing container
   echo "🛑 Stopping existing container..."
   docker stop $APP_NAME || true
   docker rm $APP_NAME || true

   # Run new container with strict resource limits for t2.micro
   echo "🏃 Starting new container..."
   docker run -d \
     --name $APP_NAME \
     --restart unless-stopped \
     -p $GRPC_PORT:50051 \
     -p $HTTP_PORT:8000 \
     --memory=600m \
     --cpus=0.7 \
     --env-file .env.production \
     -v $(pwd)/data/models:/app/models:ro \
     $DOCKER_IMAGE

   # Wait for health check
   echo "🔍 Waiting for health check..."
   for i in {1..30}; do
     if curl -f http://localhost:$HTTP_PORT/health > /dev/null 2>&1; then
       echo "✅ Service is healthy!"
       break
     fi
     echo "⏳ Waiting for service to be ready... ($i/30)"
     sleep 3
   done

   # Test gRPC service
   echo "🧪 Testing gRPC service..."
   # Add gRPC health check here

   echo "🎉 Deployment completed!"
   echo "📊 HTTP API: http://localhost:$HTTP_PORT"
   echo "🔌 gRPC Service: localhost:$GRPC_PORT"
   ```

### Deliverables

- [ ] Docker configuration optimized for t2.micro
- [ ] Deployment scripts created
- [ ] Service management configured

---

## Migration Strategy to Paid Services

### Phase 1: Preparation (Month 10-11)

1. **Model Optimization for SageMaker**

   ```python
   # deployment/scripts/prepare-sagemaker.py
   import boto3
   import tarfile
   import os

   def prepare_model_for_sagemaker():
       """Prepare local model for SageMaker deployment."""

       # Package model artifacts
       with tarfile.open('model.tar.gz', 'w:gz') as tar:
           tar.add('data/models/lstm_model.h5', arcname='model.h5')
           tar.add('data/models/scaler.pkl', arcname='scaler.pkl')

       # Upload to S3
       s3 = boto3.client('s3')
       s3.upload_file('model.tar.gz', 'your-bucket', 'models/model.tar.gz')

       print("Model prepared for SageMaker deployment")
   ```

2. **Configuration Updates**
   ```python
   # src/config/production.py
   class ProductionSettings(Settings):
       # SageMaker configuration
       sagemaker_endpoint: str = ""
       sagemaker_model_name: str = "lstm-rul-model"

       # ECS Fargate configuration
       max_workers: int = 4  # Increased for Fargate
       memory_limit_mb: int = 2048  # 2GB for Fargate

       # Redis cache for production
       redis_host: str = ""
       redis_port: int = 6379

       # Enhanced monitoring
       enable_xray: bool = True
       enable_detailed_metrics: bool = True
   ```

### Phase 2: Migration Execution (Month 12)

1. **Deploy to ECS Fargate**

   - [ ] Use existing infrastructure plan for ECS deployment
   - [ ] Configure auto-scaling for ML service
   - [ ] Set up Application Load Balancer

2. **Migrate to SageMaker**
   - [ ] Deploy model to SageMaker endpoint
   - [ ] Update inference code to use SageMaker
   - [ ] Configure model monitoring

---

## Success Metrics (Free Tier)

### Performance Targets

- **Prediction Latency**: < 2 seconds (CPU inference)
- **Memory Usage**: < 600MB (t2.micro limit)
- **CPU Usage**: < 80% (sustained)
- **Availability**: > 95%

### Operational Targets

- **Deployment Time**: < 10 minutes
- **Model Loading Time**: < 30 seconds
- **Cost**: $0/month (within free tier)

---

## Timeline Summary

| Phase                    | Duration | Dependencies       |
| ------------------------ | -------- | ------------------ |
| Phase 1: Project Setup   | 2-3 days | Python Environment |
| Phase 2: Data Processing | 4-5 days | Phase 1            |
| Phase 3: gRPC Server     | 3-4 days | Phase 2            |
| Phase 4: REST API        | 4-5 days | Phase 3            |
| Phase 5: Deployment      | 4-5 days | Phase 4            |

**Total Duration**: 17-22 days (3-4 weeks)

---

## Next Steps

1. **Immediate**: Set up Python project with free tier optimizations
2. **Week 1**: Implement data processing and lightweight model
3. **Week 2**: Develop gRPC server and REST API
4. **Week 3**: Deploy to EC2 and optimize performance
5. **Month 6-10**: Monitor and prepare for SageMaker migration
6. **Month 12**: Execute migration to paid services (SageMaker)
