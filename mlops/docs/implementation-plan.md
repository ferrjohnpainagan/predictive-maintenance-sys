# MLOps & SageMaker Pipeline Implementation Plan - Predictive Maintenance System

## Project Overview
Implementation plan for the complete MLOps pipeline using AWS SageMaker to manage the machine learning lifecycle for the RUL prediction model. This includes automated training, evaluation, deployment, monitoring, and retraining workflows.

## Technology Stack
- **ML Platform**: AWS SageMaker (Studio, Pipelines, Model Registry, Endpoints)
- **Workflow Orchestration**: SageMaker Pipelines
- **Model Storage**: S3 & Supabase Storage
- **Experiment Tracking**: SageMaker Experiments & MLflow
- **Monitoring**: SageMaker Model Monitor
- **Infrastructure**: CloudFormation/CDK
- **Languages**: Python, YAML

---

## Phase 1: SageMaker Environment Setup
**Duration**: 3-4 days  
**Priority**: Critical

### Objectives
- Set up SageMaker Studio domain
- Configure IAM roles and permissions
- Initialize project structure

### Tasks
1. **SageMaker Studio Domain Setup**
   ```python
   # infrastructure/sagemaker_domain.py
   import boto3
   from typing import Dict
   
   def create_sagemaker_domain(config: Dict):
       sagemaker = boto3.client('sagemaker')
       
       # Create execution role
       execution_role = create_sagemaker_execution_role()
       
       # Create SageMaker Studio domain
       response = sagemaker.create_domain(
           DomainName='predictive-maintenance-domain',
           AuthMode='IAM',
           DefaultUserSettings={
               'ExecutionRole': execution_role,
               'SecurityGroups': [config['security_group_id']],
               'SharingSettings': {
                   'NotebookOutputOption': 'Allowed',
                   'S3OutputPath': f"s3://{config['bucket']}/studio-output"
               }
           },
           SubnetIds=config['subnet_ids'],
           VpcId=config['vpc_id']
       )
       
       return response['DomainArn']
   ```

2. **IAM Roles Configuration**
   ```json
   {
     "SageMakerExecutionRole": {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Principal": {
             "Service": "sagemaker.amazonaws.com"
           },
           "Action": "sts:AssumeRole"
         }
       ]
     },
     "AttachedPolicies": [
       "AmazonSageMakerFullAccess",
       "AmazonS3FullAccess",
       "AmazonEC2ContainerRegistryFullAccess"
     ]
   }
   ```

3. **Project Structure**
   ```
   mlops/
   ├── pipelines/              # SageMaker Pipeline definitions
   │   ├── training/
   │   ├── evaluation/
   │   └── deployment/
   ├── processing/            # Data processing scripts
   ├── training/             # Training scripts
   ├── evaluation/           # Model evaluation scripts
   ├── monitoring/           # Model monitoring
   ├── experiments/          # Experiment tracking
   ├── infrastructure/       # CloudFormation/CDK
   └── tests/
   ```

4. **S3 Bucket Structure**
   ```
   s3://predictive-maintenance-ml/
   ├── data/
   │   ├── raw/              # Raw C-MAPSS data
   │   ├── processed/        # Preprocessed data
   │   └── train-test-split/
   ├── models/
   │   ├── artifacts/        # Trained models
   │   ├── evaluation/       # Evaluation reports
   │   └── registry/         # Model registry
   ├── pipelines/
   │   └── outputs/          # Pipeline execution outputs
   └── monitoring/
       └── baseline/         # Data drift baselines
   ```

### Deliverables
- SageMaker Studio domain configured
- IAM roles and permissions set
- S3 bucket structure created
- Initial project structure

### Testing Checklist
- [ ] SageMaker Studio accessible
- [ ] IAM roles have correct permissions
- [ ] S3 buckets created and accessible
- [ ] Can create and run notebooks

---

## Phase 2: Data Processing Pipeline
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Create data processing jobs
- Implement feature engineering pipeline
- Set up data versioning

**Note**: This phase works closely with the **Data Pipeline Implementation Plan**. Refer to that plan for detailed data ingestion, preprocessing, and quality management.

### Tasks
1. **SageMaker Processing Integration**
   - Integrate with data pipeline from **Data Pipeline Implementation Plan**
   - Configure SageMaker processing jobs to consume preprocessed data
   - Set up data versioning for ML pipeline reproducibility
   ```

2. **Processing Script**
   ```python
   # processing/preprocess.py
   import argparse
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.model_selection import train_test_split
   import joblib
   import json
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument('--sequence-length', type=int, default=50)
       parser.add_argument('--rul-ceiling', type=int, default=125)
       parser.add_argument('--test-split', type=float, default=0.2)
       args = parser.parse_args()
       
       # Load C-MAPSS data
       df = load_cmapss_data('/opt/ml/processing/input/FD001.txt')
       
       # Feature engineering
       df = add_rul_labels(df, args.rul_ceiling)
       df = select_informative_sensors(df)
       
       # Normalize features
       scaler = MinMaxScaler()
       feature_columns = [col for col in df.columns if col.startswith('s')]
       df[feature_columns] = scaler.fit_transform(df[feature_columns])
       
       # Create sequences
       X, y = create_sequences(df, args.sequence_length)
       
       # Train-test split
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=args.test_split, random_state=42
       )
       
       # Save outputs
       np.save('/opt/ml/processing/train/X_train.npy', X_train)
       np.save('/opt/ml/processing/train/y_train.npy', y_train)
       np.save('/opt/ml/processing/test/X_test.npy', X_test)
       np.save('/opt/ml/processing/test/y_test.npy', y_test)
       
       # Save scaler and metadata
       joblib.dump(scaler, '/opt/ml/processing/metadata/scaler.pkl')
       
       metadata = {
           'n_features': X_train.shape[2],
           'sequence_length': args.sequence_length,
           'n_train_samples': len(X_train),
           'n_test_samples': len(X_test),
           'feature_columns': feature_columns
       }
       
       with open('/opt/ml/processing/metadata/metadata.json', 'w') as f:
           json.dump(metadata, f)
   
   if __name__ == '__main__':
       main()
   ```

3. **Data Version Control**
   ```python
   # processing/data_versioning.py
   import hashlib
   from datetime import datetime
   
   class DataVersionManager:
       def __init__(self, s3_client):
           self.s3 = s3_client
           self.bucket = 'predictive-maintenance-ml'
       
       def create_dataset_version(self, data_path, metadata):
           # Calculate data hash
           data_hash = self.calculate_hash(data_path)
           
           # Create version tag
           version = {
               'version_id': data_hash[:8],
               'created_at': datetime.utcnow().isoformat(),
               'data_path': data_path,
               'metadata': metadata
           }
           
           # Store version info
           self.s3.put_object(
               Bucket=self.bucket,
               Key=f'data/versions/{version["version_id"]}.json',
               Body=json.dumps(version)
           )
           
           return version['version_id']
   ```

### Deliverables
- Data processing pipeline
- Feature engineering scripts
- Data versioning system
- Preprocessed datasets

### Testing Checklist
- [ ] Processing job runs successfully
- [ ] Data correctly preprocessed
- [ ] Train-test split created
- [ ] Metadata saved correctly

---

## Phase 3: Model Training Pipeline
**Duration**: 5-6 days  
**Priority**: Critical

### Objectives
- Implement SageMaker training jobs
- Create hyperparameter tuning
- Set up experiment tracking

### Tasks
1. **Training Job Configuration**
   ```python
   # training/train_job.py
   from sagemaker.tensorflow import TensorFlow
   from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
   
   class ModelTrainer:
       def __init__(self, role, instance_type='ml.p3.2xlarge'):
           self.role = role
           self.instance_type = instance_type
           
       def create_estimator(self):
           return TensorFlow(
               entry_point='train.py',
               source_dir='training',
               role=self.role,
               instance_type=self.instance_type,
               instance_count=1,
               framework_version='2.11',
               py_version='py39',
               hyperparameters={
                   'epochs': 100,
                   'batch_size': 256,
                   'learning_rate': 0.001,
                   'lstm_units_1': 64,
                   'lstm_units_2': 64,
                   'dropout_rate': 0.2,
                   'sequence_length': 50,
                   'n_features': 14
               },
               metric_definitions=[
                   {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                   {'Name': 'val:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
                   {'Name': 'val:rmse', 'Regex': 'Val RMSE: ([0-9\\.]+)'}
               ]
           )
       
       def create_tuner(self, estimator):
           hyperparameter_ranges = {
               'lstm_units_1': IntegerParameter(32, 128),
               'lstm_units_2': IntegerParameter(32, 128),
               'dropout_rate': ContinuousParameter(0.1, 0.5),
               'learning_rate': ContinuousParameter(0.0001, 0.01),
               'batch_size': IntegerParameter(128, 512)
           }
           
           return HyperparameterTuner(
               estimator,
               objective_metric_name='val:rmse',
               objective_type='Minimize',
               hyperparameter_ranges=hyperparameter_ranges,
               max_jobs=20,
               max_parallel_jobs=4,
               strategy='Bayesian'
           )
   ```

2. **Training Script**
   ```python
   # training/train.py
   import tensorflow as tf
   from tensorflow import keras
   import argparse
   import os
   import json
   
   def build_lstm_model(args):
       model = keras.Sequential([
           keras.layers.LSTM(
               args.lstm_units_1,
               return_sequences=True,
               input_shape=(args.sequence_length, args.n_features)
           ),
           keras.layers.Dropout(args.dropout_rate),
           keras.layers.LSTM(args.lstm_units_2),
           keras.layers.Dropout(args.dropout_rate),
           keras.layers.Dense(32, activation='relu'),
           keras.layers.Dropout(args.dropout_rate),
           keras.layers.Dense(1)
       ])
       
       model.compile(
           optimizer=keras.optimizers.Adam(args.learning_rate),
           loss='mse',
           metrics=['mae']
       )
       
       return model
   
   def main():
       parser = argparse.ArgumentParser()
       
       # Hyperparameters
       parser.add_argument('--epochs', type=int, default=100)
       parser.add_argument('--batch_size', type=int, default=256)
       parser.add_argument('--learning_rate', type=float, default=0.001)
       parser.add_argument('--lstm_units_1', type=int, default=64)
       parser.add_argument('--lstm_units_2', type=int, default=64)
       parser.add_argument('--dropout_rate', type=float, default=0.2)
       parser.add_argument('--sequence_length', type=int, default=50)
       parser.add_argument('--n_features', type=int, default=14)
       
       # SageMaker specific arguments
       parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
       parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
       
       args = parser.parse_args()
       
       # Load data
       X_train = np.load(os.path.join(args.train, 'X_train.npy'))
       y_train = np.load(os.path.join(args.train, 'y_train.npy'))
       X_test = np.load(os.path.join(args.test, 'X_test.npy'))
       y_test = np.load(os.path.join(args.test, 'y_test.npy'))
       
       # Build model
       model = build_lstm_model(args)
       
       # Train model
       history = model.fit(
           X_train, y_train,
           validation_data=(X_test, y_test),
           epochs=args.epochs,
           batch_size=args.batch_size,
           callbacks=[
               keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(patience=5)
           ]
       )
       
       # Save model
       model.save(os.path.join(args.model_dir, 'model.h5'))
       
       # Save training history
       with open(os.path.join(args.model_dir, 'history.json'), 'w') as f:
           json.dump(history.history, f)
   
   if __name__ == '__main__':
       main()
   ```

3. **Experiment Tracking**
   ```python
   # experiments/experiment_tracker.py
   from sagemaker.experiments import Experiment, Trial, TrialComponent
   import mlflow
   import mlflow.sagemaker
   
   class ExperimentTracker:
       def __init__(self, experiment_name):
           self.experiment = Experiment.create(
               experiment_name=experiment_name,
               description="RUL prediction model experiments"
           )
           mlflow.set_experiment(experiment_name)
       
       def create_trial(self, trial_name, parameters):
           trial = Trial.create(
               trial_name=trial_name,
               experiment_name=self.experiment.experiment_name
           )
           
           # Log parameters to MLflow
           with mlflow.start_run():
               mlflow.log_params(parameters)
           
           return trial
       
       def log_metrics(self, metrics):
           for key, value in metrics.items():
               mlflow.log_metric(key, value)
   ```

### Deliverables
- Training job configuration
- LSTM model training script
- Hyperparameter tuning setup
- Experiment tracking integration

### Testing Checklist
- [ ] Training job executes
- [ ] Model trains successfully
- [ ] Hyperparameter tuning works
- [ ] Experiments tracked properly

---

## Phase 4: Model Evaluation Pipeline
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Implement comprehensive evaluation
- Create evaluation reports
- Set quality gates

### Tasks
1. **Evaluation Job**
   ```python
   # evaluation/evaluate_job.py
   from sagemaker.processing import ScriptProcessor
   import numpy as np
   
   class ModelEvaluator:
       def __init__(self, role):
           self.processor = ScriptProcessor(
               image_uri='tensorflow/tensorflow:2.11-gpu-py39',
               role=role,
               instance_type='ml.m5.xlarge',
               instance_count=1,
               base_job_name='rul-model-evaluation'
           )
       
       def run_evaluation(self, model_path, test_data_path, output_path):
           self.processor.run(
               code='evaluation/evaluate.py',
               inputs=[
                   ProcessingInput(source=model_path, destination='/opt/ml/processing/model'),
                   ProcessingInput(source=test_data_path, destination='/opt/ml/processing/test')
               ],
               outputs=[
                   ProcessingOutput(source='/opt/ml/processing/evaluation', destination=output_path)
               ]
           )
   ```

2. **Evaluation Script**
   ```python
   # evaluation/evaluate.py
   import tensorflow as tf
   import numpy as np
   import json
   from sklearn.metrics import mean_squared_error, mean_absolute_error
   import matplotlib.pyplot as plt
   
   def calculate_phm08_score(y_true, y_pred):
       """PHM08 Challenge scoring function"""
       d = y_pred - y_true
       score = np.sum(np.where(d < 0, 
                               np.exp(-d/13) - 1,
                               np.exp(d/10) - 1))
       return score
   
   def evaluate_model(model_path, X_test, y_test):
       # Load model
       model = tf.keras.models.load_model(model_path)
       
       # Make predictions
       y_pred = model.predict(X_test).flatten()
       
       # Calculate metrics
       metrics = {
           'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
           'mae': mean_absolute_error(y_test, y_pred),
           'phm08_score': calculate_phm08_score(y_test, y_pred),
           'mean_prediction': float(np.mean(y_pred)),
           'std_prediction': float(np.std(y_pred)),
           'min_prediction': float(np.min(y_pred)),
           'max_prediction': float(np.max(y_pred))
       }
       
       # Create visualizations
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       # Predictions vs Actual
       axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
       axes[0, 0].plot([0, max(y_test)], [0, max(y_test)], 'r--')
       axes[0, 0].set_xlabel('Actual RUL')
       axes[0, 0].set_ylabel('Predicted RUL')
       axes[0, 0].set_title('Predictions vs Actual')
       
       # Error distribution
       errors = y_pred - y_test
       axes[0, 1].hist(errors, bins=50, edgecolor='black')
       axes[0, 1].set_xlabel('Prediction Error')
       axes[0, 1].set_ylabel('Frequency')
       axes[0, 1].set_title('Error Distribution')
       
       # RUL distribution
       axes[1, 0].hist([y_test, y_pred], label=['Actual', 'Predicted'], 
                      bins=30, alpha=0.7)
       axes[1, 0].set_xlabel('RUL')
       axes[1, 0].set_ylabel('Frequency')
       axes[1, 0].set_title('RUL Distribution')
       axes[1, 0].legend()
       
       # Error by RUL range
       rul_ranges = [(0, 20), (20, 50), (50, 100), (100, 125)]
       range_errors = []
       range_labels = []
       
       for low, high in rul_ranges:
           mask = (y_test >= low) & (y_test < high)
           if np.any(mask):
               range_errors.append(errors[mask])
               range_labels.append(f'{low}-{high}')
       
       axes[1, 1].boxplot(range_errors, labels=range_labels)
       axes[1, 1].set_xlabel('RUL Range')
       axes[1, 1].set_ylabel('Prediction Error')
       axes[1, 1].set_title('Error by RUL Range')
       
       plt.tight_layout()
       plt.savefig('/opt/ml/processing/evaluation/evaluation_plots.png')
       
       return metrics
   
   def main():
       # Load test data
       X_test = np.load('/opt/ml/processing/test/X_test.npy')
       y_test = np.load('/opt/ml/processing/test/y_test.npy')
       
       # Evaluate model
       metrics = evaluate_model('/opt/ml/processing/model/model.h5', X_test, y_test)
       
       # Quality gates
       quality_gates = {
           'rmse_threshold': 20,
           'phm08_threshold': 500,
           'passed': metrics['rmse'] < 20 and metrics['phm08_score'] < 500
       }
       
       # Save evaluation report
       report = {
           'metrics': metrics,
           'quality_gates': quality_gates,
           'timestamp': datetime.utcnow().isoformat()
       }
       
       with open('/opt/ml/processing/evaluation/report.json', 'w') as f:
           json.dump(report, f, indent=2)
       
       print(f"Evaluation complete. Metrics: {metrics}")
       print(f"Quality gates passed: {quality_gates['passed']}")
   
   if __name__ == '__main__':
       main()
   ```

### Deliverables
- Model evaluation pipeline
- Comprehensive metrics calculation
- Visualization generation
- Quality gate implementation

### Testing Checklist
- [ ] Evaluation runs successfully
- [ ] Metrics calculated correctly
- [ ] Visualizations generated
- [ ] Quality gates enforced

---

## Phase 5: Model Registry and Deployment
**Duration**: 4-5 days  
**Priority**: Critical

### Objectives
- Set up model registry
- Implement deployment pipeline
- Configure endpoints

### Tasks
1. **Model Registry**
   ```python
   # deployment/model_registry.py
   from sagemaker.model import Model
   from sagemaker.model_metrics import ModelMetrics
   import boto3
   
   class ModelRegistry:
       def __init__(self, role):
           self.role = role
           self.sm_client = boto3.client('sagemaker')
           self.model_package_group = 'rul-prediction-models'
       
       def create_model_package_group(self):
           try:
               self.sm_client.create_model_package_group(
                   ModelPackageGroupName=self.model_package_group,
                   ModelPackageGroupDescription='RUL prediction models'
               )
           except:
               pass  # Group already exists
       
       def register_model(self, model_artifacts, evaluation_report):
           # Create model metrics
           model_metrics = ModelMetrics(
               model_statistics=MetricsSource(
                   s3_uri=evaluation_report,
                   content_type='application/json'
               )
           )
           
           # Register model
           model_package = self.sm_client.create_model_package(
               ModelPackageGroupName=self.model_package_group,
               ModelPackageDescription='LSTM RUL prediction model',
               InferenceSpecification={
                   'Containers': [{
                       'Image': 'tensorflow/tensorflow:2.11-gpu',
                       'ModelDataUrl': model_artifacts,
                       'Framework': 'TENSORFLOW'
                   }],
                   'SupportedContentTypes': ['application/json'],
                   'SupportedResponseMIMETypes': ['application/json']
               },
               ModelMetrics=model_metrics,
               ModelApprovalStatus='PendingManualApproval'
           )
           
           return model_package['ModelPackageArn']
       
       def approve_model(self, model_package_arn):
           self.sm_client.update_model_package(
               ModelPackageArn=model_package_arn,
               ModelApprovalStatus='Approved'
           )
   ```

2. **Endpoint Deployment**
   ```python
   # deployment/deploy_endpoint.py
   from sagemaker.tensorflow import TensorFlowModel
   from sagemaker.serializers import JSONSerializer
   from sagemaker.deserializers import JSONDeserializer
   
   class EndpointDeployer:
       def __init__(self, role):
           self.role = role
           
       def deploy_model(self, model_package_arn, endpoint_name):
           model = TensorFlowModel(
               model_data=model_package_arn,
               role=self.role,
               framework_version='2.11'
           )
           
           predictor = model.deploy(
               initial_instance_count=1,
               instance_type='ml.m5.xlarge',
               endpoint_name=endpoint_name,
               serializer=JSONSerializer(),
               deserializer=JSONDeserializer()
           )
           
           return predictor
       
       def create_multi_variant_endpoint(self, models, endpoint_name):
           """Deploy A/B testing endpoint"""
           from sagemaker.model import Model
           from sagemaker.pipeline import PipelineModel
           
           # Create endpoint config with multiple variants
           endpoint_config = self.sm_client.create_endpoint_config(
               EndpointConfigName=f'{endpoint_name}-config',
               ProductionVariants=[
                   {
                       'VariantName': 'variant-a',
                       'ModelName': models[0],
                       'InitialInstanceCount': 1,
                       'InstanceType': 'ml.m5.xlarge',
                       'InitialVariantWeight': 0.5
                   },
                   {
                       'VariantName': 'variant-b',
                       'ModelName': models[1],
                       'InitialInstanceCount': 1,
                       'InstanceType': 'ml.m5.xlarge',
                       'InitialVariantWeight': 0.5
                   }
               ]
           )
           
           # Create endpoint
           endpoint = self.sm_client.create_endpoint(
               EndpointName=endpoint_name,
               EndpointConfigName=f'{endpoint_name}-config'
           )
           
           return endpoint
   ```

3. **Inference Configuration**
   ```python
   # deployment/inference.py
   import json
   import tensorflow as tf
   
   def model_fn(model_dir):
       """Load model for inference"""
       model = tf.keras.models.load_model(f'{model_dir}/model.h5')
       return model
   
   def input_fn(request_body, request_content_type):
       """Parse input data"""
       if request_content_type == 'application/json':
           input_data = json.loads(request_body)
           return np.array(input_data['instances'])
       else:
           raise ValueError(f'Unsupported content type: {request_content_type}')
   
   def predict_fn(input_data, model):
       """Make predictions"""
       predictions = model.predict(input_data)
       return predictions
   
   def output_fn(prediction, content_type):
       """Format output"""
       if content_type == 'application/json':
           return json.dumps({
               'predictions': prediction.tolist()
           })
       else:
           raise ValueError(f'Unsupported content type: {content_type}')
   ```

### Deliverables
- Model registry setup
- Deployment pipeline
- Endpoint configuration
- A/B testing capability

### Testing Checklist
- [ ] Models register correctly
- [ ] Endpoints deploy successfully
- [ ] Inference works
- [ ] A/B testing configured

---

## Phase 6: Automated Pipeline Orchestration
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Create end-to-end pipeline
- Implement conditional logic
- Set up triggers

### Tasks
1. **SageMaker Pipeline Definition**
   ```python
   # pipelines/training_pipeline.py
   from sagemaker.workflow.pipeline import Pipeline
   from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
   from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
   from sagemaker.workflow.condition_step import ConditionStep
   from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat
   
   class RULPipeline:
       def __init__(self, role, bucket):
           self.role = role
           self.bucket = bucket
           
           # Pipeline parameters
           self.sequence_length = ParameterInteger(name='SequenceLength', default_value=50)
           self.lstm_units = ParameterInteger(name='LSTMUnits', default_value=64)
           self.rmse_threshold = ParameterFloat(name='RMSEThreshold', default_value=20.0)
       
       def create_pipeline(self):
           # Step 1: Data Processing
           processing_step = ProcessingStep(
               name='DataProcessing',
               processor=self.create_processor(),
               inputs=[...],
               outputs=[...],
               code='processing/preprocess.py'
           )
           
           # Step 2: Model Training
           training_step = TrainingStep(
               name='ModelTraining',
               estimator=self.create_estimator(),
               inputs={
                   'train': TrainingInput(
                       s3_data=processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri
                   )
               }
           )
           
           # Step 3: Model Evaluation
           evaluation_step = ProcessingStep(
               name='ModelEvaluation',
               processor=self.create_evaluator(),
               inputs=[
                   ProcessingInput(
                       source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                       destination='/opt/ml/processing/model'
                   )
               ],
               outputs=[
                   ProcessingOutput(
                       output_name='evaluation',
                       source='/opt/ml/processing/evaluation'
                   )
               ],
               code='evaluation/evaluate.py'
           )
           
           # Step 4: Register Model (conditional)
           register_step = CreateModelStep(
               name='RegisterModel',
               model=Model(
                   image_uri='tensorflow/tensorflow:2.11-gpu',
                   model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                   role=self.role
               )
           )
           
           # Condition: Only register if RMSE < threshold
           condition = ConditionLessThanOrEqualTo(
               left=evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
               right=self.rmse_threshold
           )
           
           condition_step = ConditionStep(
               name='CheckModelQuality',
               conditions=[condition],
               if_steps=[register_step],
               else_steps=[]
           )
           
           # Create pipeline
           pipeline = Pipeline(
               name='rul-prediction-pipeline',
               parameters=[
                   self.sequence_length,
                   self.lstm_units,
                   self.rmse_threshold
               ],
               steps=[
                   processing_step,
                   training_step,
                   evaluation_step,
                   condition_step
               ]
           )
           
           return pipeline
   ```

2. **Pipeline Triggers**
   ```python
   # pipelines/triggers.py
   import boto3
   from datetime import datetime
   
   class PipelineTrigger:
       def __init__(self):
           self.sm_client = boto3.client('sagemaker')
           self.events_client = boto3.client('events')
       
       def create_schedule_trigger(self, pipeline_name, schedule='rate(7 days)'):
           """Create scheduled pipeline execution"""
           rule_name = f'{pipeline_name}-schedule'
           
           # Create EventBridge rule
           self.events_client.put_rule(
               Name=rule_name,
               ScheduleExpression=schedule,
               State='ENABLED'
           )
           
           # Add SageMaker pipeline as target
           self.events_client.put_targets(
               Rule=rule_name,
               Targets=[{
                   'Arn': f'arn:aws:sagemaker:region:account:pipeline/{pipeline_name}',
                   'RoleArn': self.role_arn,
                   'SageMakerPipelineParameters': {
                       'PipelineParameterList': []
                   }
               }]
           )
       
       def create_data_trigger(self, pipeline_name, s3_prefix):
           """Trigger pipeline on new data arrival"""
           # S3 event notification -> Lambda -> SageMaker Pipeline
           lambda_function = f'''
           import boto3
           
           def lambda_handler(event, context):
               sm_client = boto3.client('sagemaker')
               
               # Start pipeline execution
               response = sm_client.start_pipeline_execution(
                   PipelineName='{pipeline_name}',
                   PipelineExecutionDisplayName=f'triggered-{datetime.now()}'
               )
               
               return response
           '''
           
           # Configure S3 bucket notification
           s3_client = boto3.client('s3')
           s3_client.put_bucket_notification_configuration(
               Bucket=self.bucket,
               NotificationConfiguration={
                   'LambdaFunctionConfigurations': [{
                       'LambdaFunctionArn': lambda_arn,
                       'Events': ['s3:ObjectCreated:*'],
                       'Filter': {
                           'Key': {
                               'FilterRules': [{
                                   'Name': 'prefix',
                                   'Value': s3_prefix
                               }]
                           }
                       }
                   }]
               }
           )
   ```

### Deliverables
- Complete pipeline definition
- Conditional execution logic
- Automated triggers
- Pipeline monitoring

### Testing Checklist
- [ ] Pipeline executes end-to-end
- [ ] Conditional logic works
- [ ] Triggers function correctly
- [ ] Pipeline monitoring active

---

## Phase 7: Model Monitoring and Drift Detection
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Implement model monitoring
- Set up drift detection
- Create alerting system

**Note**: This phase integrates with the **Monitoring & Observability Implementation Plan** and **Data Pipeline Implementation Plan** for comprehensive monitoring coverage.

### Tasks
1. **SageMaker Model Monitor Setup**
   - Configure SageMaker Model Monitor for endpoint monitoring
   - Enable data capture for inference requests
   - Set up baseline creation from training data

2. **Drift Detection Integration**
   - Integrate with data quality monitoring from **Data Pipeline Implementation Plan**
   - Combine model drift detection with data drift detection
   - Use drift metrics from **Monitoring & Observability Implementation Plan**

3. **Alerting Integration**
   - Leverage alerting framework from **Monitoring & Observability Implementation Plan**
   - Configure SageMaker-specific alerts for model performance
   - Set up escalation procedures for model degradation
   ```

2. **Drift Detection**
   ```python
   # monitoring/drift_detector.py
   import pandas as pd
   from scipy import stats
   
   class DriftDetector:
       def __init__(self, baseline_stats):
           self.baseline_stats = baseline_stats
           self.drift_threshold = 0.05  # p-value threshold
       
       def detect_data_drift(self, current_data):
           """Detect distribution drift using KS test"""
           drift_results = {}
           
           for feature in current_data.columns:
               if feature in self.baseline_stats:
                   baseline_dist = self.baseline_stats[feature]
                   current_dist = current_data[feature]
                   
                   # Kolmogorov-Smirnov test
                   ks_statistic, p_value = stats.ks_2samp(
                       baseline_dist, current_dist
                   )
                   
                   drift_results[feature] = {
                       'ks_statistic': ks_statistic,
                       'p_value': p_value,
                       'drift_detected': p_value < self.drift_threshold
                   }
           
           return drift_results
       
       def detect_prediction_drift(self, recent_predictions, baseline_predictions):
           """Detect drift in model predictions"""
           # Population Stability Index (PSI)
           psi = self.calculate_psi(baseline_predictions, recent_predictions)
           
           drift_detected = psi > 0.2  # PSI > 0.2 indicates significant drift
           
           return {
               'psi': psi,
               'drift_detected': drift_detected,
               'severity': 'high' if psi > 0.25 else 'medium' if psi > 0.1 else 'low'
           }
   ```

3. **Alerting System**
   ```python
   # monitoring/alerts.py
   import boto3
   
   class AlertManager:
       def __init__(self):
           self.sns_client = boto3.client('sns')
           self.cloudwatch = boto3.client('cloudwatch')
           self.topic_arn = 'arn:aws:sns:region:account:model-monitoring-alerts'
       
       def create_alarms(self, endpoint_name):
           # Model latency alarm
           self.cloudwatch.put_metric_alarm(
               AlarmName=f'{endpoint_name}-high-latency',
               ComparisonOperator='GreaterThanThreshold',
               EvaluationPeriods=2,
               MetricName='ModelLatency',
               Namespace='AWS/SageMaker',
               Period=300,
               Statistic='Average',
               Threshold=200.0,
               ActionsEnabled=True,
               AlarmActions=[self.topic_arn],
               AlarmDescription='Alert when model latency exceeds 200ms'
           )
           
           # Drift detection alarm
           self.cloudwatch.put_metric_alarm(
               AlarmName=f'{endpoint_name}-data-drift',
               ComparisonOperator='GreaterThanThreshold',
               EvaluationPeriods=1,
               MetricName='DataDrift',
               Namespace='CustomMetrics/ModelMonitoring',
               Period=3600,
               Statistic='Maximum',
               Threshold=1.0,
               ActionsEnabled=True,
               AlarmActions=[self.topic_arn]
           )
       
       def send_alert(self, alert_type, details):
           message = f'''
           Model Monitoring Alert: {alert_type}
           
           Details:
           {json.dumps(details, indent=2)}
           
           Action Required: Review model performance and consider retraining.
           '''
           
           self.sns_client.publish(
               TopicArn=self.topic_arn,
               Subject=f'Model Alert: {alert_type}',
               Message=message
           )
   ```

### Deliverables
- Model monitoring setup
- Drift detection system
- Alerting mechanism
- Monitoring dashboard

### Testing Checklist
- [ ] Data capture enabled
- [ ] Baseline created
- [ ] Monitoring schedule active
- [ ] Alerts trigger correctly

---

## Phase 8: Production Operations and Maintenance
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Implement rollback procedures
- Create disaster recovery
- Set up operational dashboards

### Tasks
1. **Rollback Procedures**
   ```python
   # operations/rollback.py
   class ModelRollback:
       def __init__(self):
           self.sm_client = boto3.client('sagemaker')
       
       def rollback_endpoint(self, endpoint_name, previous_config):
           """Rollback to previous endpoint configuration"""
           # Update endpoint with previous configuration
           self.sm_client.update_endpoint(
               EndpointName=endpoint_name,
               EndpointConfigName=previous_config
           )
           
           # Wait for update to complete
           waiter = self.sm_client.get_waiter('endpoint_in_service')
           waiter.wait(EndpointName=endpoint_name)
       
       def create_endpoint_backup(self, endpoint_name):
           """Create backup of current endpoint configuration"""
           response = self.sm_client.describe_endpoint(
               EndpointName=endpoint_name
           )
           
           backup = {
               'endpoint_config': response['EndpointConfigName'],
               'timestamp': datetime.utcnow().isoformat(),
               'model_data': self.get_model_artifacts(response['EndpointConfigName'])
           }
           
           # Store backup
           s3_client = boto3.client('s3')
           s3_client.put_object(
               Bucket=self.bucket,
               Key=f'backups/{endpoint_name}/{backup["timestamp"]}.json',
               Body=json.dumps(backup)
           )
   ```

2. **Operational Dashboard**
   ```python
   # operations/dashboard.py
   import streamlit as st
   import pandas as pd
   import plotly.graph_objects as go
   
   def create_mlops_dashboard():
       st.title('MLOps Dashboard - RUL Prediction System')
       
       # Model Performance Metrics
       st.header('Model Performance')
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric('Current RMSE', '18.5', '-1.2')
       with col2:
           st.metric('PHM08 Score', '420', '-30')
       with col3:
           st.metric('Avg Latency', '95ms', '+5ms')
       with col4:
           st.metric('Daily Predictions', '1,250', '+150')
       
       # Drift Detection
       st.header('Drift Monitoring')
       drift_data = pd.DataFrame({
           'Feature': ['s2', 's3', 's4', 's7'],
           'PSI': [0.05, 0.08, 0.12, 0.22],
           'Status': ['OK', 'OK', 'Warning', 'Alert']
       })
       
       fig = go.Figure(data=[
           go.Bar(x=drift_data['Feature'], y=drift_data['PSI'])
       ])
       fig.add_hline(y=0.2, line_dash="dash", line_color="red")
       st.plotly_chart(fig)
       
       # Pipeline Executions
       st.header('Pipeline Status')
       pipeline_status = pd.DataFrame({
           'Pipeline': ['Training', 'Evaluation', 'Deployment'],
           'Last Run': ['2024-01-15 10:00', '2024-01-15 11:30', '2024-01-15 12:00'],
           'Status': ['Success', 'Success', 'In Progress'],
           'Duration': ['45 min', '15 min', '-']
       })
       st.dataframe(pipeline_status)
   ```

### Deliverables
- Rollback procedures
- Backup strategies
- Operational dashboard
- Disaster recovery plan

### Testing Checklist
- [ ] Rollback procedures work
- [ ] Backups created successfully
- [ ] Dashboard displays metrics
- [ ] Recovery procedures tested

---

## Success Metrics

### Pipeline Performance
- Training pipeline execution: < 2 hours
- Model deployment time: < 15 minutes
- Rollback time: < 5 minutes
- Pipeline success rate: > 95%

### Model Quality
- RMSE maintained: < 20 cycles
- Drift detection accuracy: > 90%
- False positive rate: < 5%

### Operational Excellence
- Pipeline automation: 100%
- Monitoring coverage: 100%
- Alert response time: < 5 minutes

---

## Risk Mitigation

### Technical Risks
1. **Pipeline Failures**
   - Mitigation: Implement retry logic and fallback mechanisms

2. **Model Degradation**
   - Mitigation: Automated retraining and A/B testing

3. **Data Quality Issues**
   - Mitigation: Data validation and quality checks

### Operational Risks
1. **Cost Overruns**
   - Mitigation: Cost monitoring and budget alerts

2. **Security Breaches**
   - Mitigation: IAM policies and encryption

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Environment Setup | 3-4 days | AWS Account |
| Phase 2: Data Processing | 4-5 days | Phase 1 |
| Phase 3: Model Training | 5-6 days | Phase 2 |
| Phase 4: Model Evaluation | 3-4 days | Phase 3 |
| Phase 5: Registry & Deployment | 4-5 days | Phase 4 |
| Phase 6: Pipeline Orchestration | 4-5 days | Phase 1-5 |
| Phase 7: Monitoring | 3-4 days | Phase 5 |
| Phase 8: Operations | 3-4 days | All phases |

**Total Duration**: 29-37 days (6-7 weeks)

---

## Next Steps
1. Set up AWS account and permissions
2. Create SageMaker Studio domain
3. Prepare C-MAPSS dataset
4. Begin Phase 1 implementation
5. Coordinate with ML service team for model artifacts