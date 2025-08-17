# Data Pipeline Implementation Plan - Predictive Maintenance System

## Project Overview
Implementation plan for the complete data pipeline that handles C-MAPSS dataset ingestion, preprocessing, continuous sensor data simulation, and data quality management for the predictive maintenance system.

## Technology Stack
- **ETL Processing**: Apache Airflow (on AWS MWAA)
- **Data Storage**: Supabase Postgres, AWS S3
- **Stream Processing**: AWS Kinesis Data Streams
- **Data Quality**: Great Expectations
- **Orchestration**: AWS Step Functions
- **Monitoring**: AWS CloudWatch, DataDog

---

## Phase 1: Data Architecture Setup
**Duration**: 3-4 days  
**Priority**: Critical

### Objectives
- Design data flow architecture
- Set up storage layers
- Configure data governance

### Tasks
1. **Data Lake Architecture**
   ```sql
   -- Supabase Schema Design
   -- engines table
   CREATE TABLE engines (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     unit_number VARCHAR(50) UNIQUE NOT NULL,
     model VARCHAR(100),
     manufactured_date DATE,
     first_operation_date DATE,
     status VARCHAR(20) DEFAULT 'operational',
     created_at TIMESTAMP DEFAULT NOW(),
     updated_at TIMESTAMP DEFAULT NOW()
   );
   
   -- sensor_data table (partitioned by date)
   CREATE TABLE sensor_data (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     engine_id UUID REFERENCES engines(id),
     cycle INTEGER NOT NULL,
     timestamp TIMESTAMP NOT NULL,
     
     -- Operational Settings
     altitude REAL,           -- setting_1 (ft)
     mach_number REAL,        -- setting_2
     tra REAL,                -- setting_3 (Throttle Resolver Angle %)
     
     -- Sensor Readings
     s2 REAL,   -- T24 - Total temperature at LPC outlet (°R)
     s3 REAL,   -- T30 - Total temperature at HPC outlet (°R)
     s4 REAL,   -- T50 - Total temperature at LPT outlet (°R)
     s7 REAL,   -- P30 - Total pressure at HPC outlet (psia)
     s8 REAL,   -- Nf - Physical fan speed (rpm)
     s9 REAL,   -- Nc - Physical core speed (rpm)
     s11 REAL,  -- Ps30 - Static pressure at HPC outlet (psia)
     s12 REAL,  -- phi - Ratio of fuel flow to Ps30 (pps/psia)
     s13 REAL,  -- NRf - Corrected fan speed (rpm)
     s14 REAL,  -- NRc - Corrected core speed (rpm)
     s15 REAL,  -- BPR - Bypass Ratio
     s17 REAL,  -- htBleed - HPT coolant bleed (lbm/s)
     s20 REAL,  -- W31 - LPT coolant bleed (lbm/s)
     s21 REAL,  -- W32 - HPT coolant bleed (lbm/s)
     
     created_at TIMESTAMP DEFAULT NOW()
   ) PARTITION BY RANGE (timestamp);
   
   -- Create monthly partitions
   CREATE TABLE sensor_data_2024_01 PARTITION OF sensor_data
     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
   
   -- rul_predictions table
   CREATE TABLE rul_predictions (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     engine_id UUID REFERENCES engines(id),
     predicted_rul INTEGER NOT NULL,
     confidence REAL,
     model_version VARCHAR(50),
     prediction_timestamp TIMESTAMP DEFAULT NOW(),
     input_data_range TSTZRANGE, -- Time range of input data used
     
     created_at TIMESTAMP DEFAULT NOW()
   );
   
   -- data_quality_metrics table
   CREATE TABLE data_quality_metrics (
     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
     table_name VARCHAR(100),
     metric_name VARCHAR(100),
     metric_value REAL,
     threshold_min REAL,
     threshold_max REAL,
     status VARCHAR(20), -- 'pass', 'warning', 'fail'
     check_timestamp TIMESTAMP DEFAULT NOW()
   );
   ```

2. **S3 Data Organization**
   ```
   s3://predictive-maintenance-data/
   ├── raw/                    # Original C-MAPSS files
   │   ├── train_FD001.txt
   │   ├── test_FD001.txt
   │   └── RUL_FD001.txt
   ├── processed/             # Cleaned and transformed data
   │   ├── year=2024/
   │   │   ├── month=01/
   │   │   │   ├── day=01/
   │   │   │   │   └── sensor_data.parquet
   │   └── train-test-split/
   │       ├── X_train.parquet
   │       ├── y_train.parquet
   │       ├── X_test.parquet
   │       └── y_test.parquet
   ├── streaming/             # Real-time streaming data
   │   ├── kinesis-firehose/
   │   └── archived/
   ├── quality/               # Data quality reports
   │   ├── expectations/
   │   └── reports/
   └── backups/              # Database backups
       └── daily/
   ```

3. **Data Governance Framework**
   ```python
   # data_governance/data_catalog.py
   from dataclasses import dataclass
   from typing import Dict, List, Optional
   from enum import Enum
   
   class DataClassification(Enum):
       PUBLIC = "public"
       INTERNAL = "internal"
       CONFIDENTIAL = "confidential"
       RESTRICTED = "restricted"
   
   class DataQuality(Enum):
       RAW = "raw"
       CLEANED = "cleaned"
       VALIDATED = "validated"
       ENRICHED = "enriched"
   
   @dataclass
   class DataAsset:
       name: str
       description: str
       owner: str
       classification: DataClassification
       quality_level: DataQuality
       schema_version: str
       retention_days: int
       tags: List[str]
       lineage: Optional[Dict]
       
   # Register data assets
   SENSOR_DATA_ASSET = DataAsset(
       name="sensor_data",
       description="Real-time sensor readings from aircraft engines",
       owner="data-engineering-team",
       classification=DataClassification.INTERNAL,
       quality_level=DataQuality.VALIDATED,
       schema_version="v1.0",
       retention_days=2555,  # 7 years
       tags=["sensors", "time-series", "rul-prediction"],
       lineage={
           "sources": ["C-MAPSS-dataset", "simulation-engine"],
           "transforms": ["data-cleaning", "feature-engineering"],
           "consumers": ["ml-training", "dashboard", "api"]
       }
   )
   ```

### Deliverables
- Database schema created
- S3 bucket structure organized
- Data governance framework defined
- Initial data catalog

### Testing Checklist
- [ ] Database schema deployed
- [ ] S3 buckets accessible
- [ ] Data governance policies active
- [ ] Schema validation working

---

## Phase 2: C-MAPSS Dataset Ingestion
**Duration**: 3-4 days  
**Priority**: Critical

### Objectives
- Import C-MAPSS dataset
- Transform and load into Supabase
- Validate data integrity

### Tasks
1. **Data Ingestion Pipeline**
   ```python
   # ingestion/cmapss_loader.py
   import pandas as pd
   import numpy as np
   from supabase import create_client
   import logging
   from typing import Dict, List
   
   class CMAPSSLoader:
       def __init__(self, supabase_url: str, supabase_key: str):
           self.supabase = create_client(supabase_url, supabase_key)
           self.logger = logging.getLogger(__name__)
           
           # Column mappings
           self.column_names = [
               'unit_number', 'time_cycles', 'altitude', 'mach_number', 'tra'
           ] + [f's{i}' for i in range(1, 22)]
           
           # Sensors to keep (remove non-informative ones)
           self.keep_sensors = [
               's2', 's3', 's4', 's7', 's8', 's9', 's11', 
               's12', 's13', 's14', 's15', 's17', 's20', 's21'
           ]
       
       def load_training_data(self, file_path: str) -> pd.DataFrame:
           """Load and parse C-MAPSS training data"""
           df = pd.read_csv(
               file_path,
               sep=' ',
               header=None,
               names=self.column_names + ['col_26', 'col_27'],  # Extra columns
               index_col=False
           )
           
           # Remove extra columns and non-informative sensors
           df = df.drop(columns=['col_26', 'col_27'] + 
                       [col for col in df.columns if col.startswith('s') and col not in self.keep_sensors])
           
           # Add RUL calculations
           df = self._add_rul_labels(df)
           
           # Add timestamps (simulated operational cycles)
           df = self._add_timestamps(df)
           
           return df
       
       def _add_rul_labels(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
           """Calculate RUL for each engine cycle"""
           max_cycles = df.groupby('unit_number')['time_cycles'].max()
           
           def calculate_rul(row):
               max_cycle = max_cycles[row['unit_number']]
               rul = max_cycle - row['time_cycles']
               return min(rul, max_rul)  # Apply ceiling
           
           df['rul'] = df.apply(calculate_rul, axis=1)
           return df
       
       def _add_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
           """Add realistic timestamps based on operational cycles"""
           from datetime import datetime, timedelta
           
           # Assume each cycle represents 1 day of operation
           base_date = datetime(2020, 1, 1)
           
           def calculate_timestamp(row):
               return base_date + timedelta(days=int(row['time_cycles']))
           
           df['timestamp'] = df.apply(calculate_timestamp, axis=1)
           return df
       
       def load_to_database(self, df: pd.DataFrame, batch_size: int = 1000):
           """Load data into Supabase in batches"""
           
           # First, create engine records
           engines_df = df[['unit_number']].drop_duplicates()
           
           for _, engine in engines_df.iterrows():
               engine_data = {
                   'unit_number': f"ENGINE_{engine['unit_number']:03d}",
                   'model': 'CFM56-7B',
                   'manufactured_date': '2019-01-01',
                   'first_operation_date': '2020-01-01',
                   'status': 'operational'
               }
               
               result = self.supabase.table('engines').insert(engine_data).execute()
               self.logger.info(f"Inserted engine {engine['unit_number']}")
           
           # Load sensor data in batches
           total_rows = len(df)
           
           for i in range(0, total_rows, batch_size):
               batch = df.iloc[i:i + batch_size]
               
               # Convert to records for insertion
               records = []
               for _, row in batch.iterrows():
                   # Get engine ID
                   engine_result = self.supabase.table('engines').select('id').eq(
                       'unit_number', f"ENGINE_{row['unit_number']:03d}"
                   ).execute()
                   
                   if engine_result.data:
                       engine_id = engine_result.data[0]['id']
                       
                       record = {
                           'engine_id': engine_id,
                           'cycle': int(row['time_cycles']),
                           'timestamp': row['timestamp'].isoformat(),
                           'altitude': float(row['altitude']),
                           'mach_number': float(row['mach_number']),
                           'tra': float(row['tra']),
                       }
                       
                       # Add sensor readings
                       for sensor in self.keep_sensors:
                           record[sensor] = float(row[sensor])
                       
                       records.append(record)
               
               # Insert batch
               if records:
                   result = self.supabase.table('sensor_data').insert(records).execute()
                   self.logger.info(f"Inserted batch {i//batch_size + 1}/{(total_rows//batch_size) + 1}")
   ```

2. **Data Validation Pipeline**
   ```python
   # validation/data_validator.py
   import great_expectations as ge
   from great_expectations.dataset import PandasDataset
   
   class DataValidator:
       def __init__(self):
           self.context = ge.DataContext()
           
       def create_expectations_suite(self, df: pd.DataFrame) -> dict:
           """Create data quality expectations"""
           
           dataset = PandasDataset(df)
           
           # Basic expectations
           dataset.expect_table_row_count_to_be_between(min_value=1000)
           dataset.expect_table_columns_to_match_ordered_list([
               'unit_number', 'time_cycles', 'altitude', 'mach_number', 'tra'
           ] + [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]])
           
           # Engine number expectations
           dataset.expect_column_values_to_be_between('unit_number', min_value=1, max_value=100)
           dataset.expect_column_values_to_not_be_null('unit_number')
           
           # Cycle expectations
           dataset.expect_column_values_to_be_of_type('time_cycles', 'int')
           dataset.expect_column_values_to_be_between('time_cycles', min_value=1, max_value=500)
           
           # Sensor value expectations (based on C-MAPSS data characteristics)
           sensor_ranges = {
               's2': (630, 650),   # T24
               's3': (1550, 1650), # T30
               's4': (1350, 1450), # T50
               's7': (2.5, 3.5),   # P30
               's8': (9000, 10000), # Nf
               's9': (8000, 9500),  # Nc
           }
           
           for sensor, (min_val, max_val) in sensor_ranges.items():
               dataset.expect_column_values_to_be_between(
                   sensor, min_value=min_val, max_value=max_val
               )
               dataset.expect_column_values_to_not_be_null(sensor)
           
           return dataset.get_expectation_suite()
       
       def validate_data(self, df: pd.DataFrame) -> dict:
           """Run validation checks"""
           suite = self.create_expectations_suite(df)
           dataset = PandasDataset(df)
           
           results = dataset.validate(expectation_suite=suite)
           
           # Log results
           if results['success']:
               self.logger.info("Data validation passed")
           else:
               failed_expectations = [
                   exp for exp in results['results'] 
                   if not exp['success']
               ]
               self.logger.warning(f"Data validation failed: {len(failed_expectations)} issues")
           
           return results
   ```

3. **Airflow DAG for Ingestion**
   ```python
   # airflow/dags/cmapss_ingestion_dag.py
   from datetime import datetime, timedelta
   from airflow import DAG
   from airflow.operators.python_operator import PythonOperator
   from airflow.operators.bash_operator import BashOperator
   
   default_args = {
       'owner': 'data-engineering',
       'depends_on_past': False,
       'start_date': datetime(2024, 1, 1),
       'email_on_failure': True,
       'email_on_retry': False,
       'retries': 2,
       'retry_delay': timedelta(minutes=5)
   }
   
   dag = DAG(
       'cmapss_data_ingestion',
       default_args=default_args,
       description='Ingest C-MAPSS dataset into Supabase',
       schedule_interval='@once',  # Run once for initial load
       catchup=False
   )
   
   def download_cmapss_data():
       """Download C-MAPSS dataset from NASA"""
       import requests
       import zipfile
       
       url = "https://ti.arc.nasa.gov/c/6/"
       # Download and extract logic here
       pass
   
   def load_and_validate():
       """Load data and run validation"""
       loader = CMAPSSLoader(
           supabase_url=Variable.get("SUPABASE_URL"),
           supabase_key=Variable.get("SUPABASE_SERVICE_KEY")
       )
       
       validator = DataValidator()
       
       # Load training data
       df = loader.load_training_data('/tmp/train_FD001.txt')
       
       # Validate
       results = validator.validate_data(df)
       
       if results['success']:
           # Load to database
           loader.load_to_database(df)
       else:
           raise ValueError("Data validation failed")
   
   # Tasks
   download_task = PythonOperator(
       task_id='download_cmapss_data',
       python_callable=download_cmapss_data,
       dag=dag
   )
   
   load_task = PythonOperator(
       task_id='load_and_validate',
       python_callable=load_and_validate,
       dag=dag
   )
   
   # Dependencies
   download_task >> load_task
   ```

### Deliverables
- C-MAPSS data ingested into Supabase
- Data validation pipeline
- Airflow DAG for orchestration
- Data quality reports

### Testing Checklist
- [ ] C-MAPSS data loaded successfully
- [ ] Data validation passes
- [ ] Airflow DAG executes
- [ ] Database queries work

---

## Phase 3: Real-time Data Simulation
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Create real-time sensor data simulation
- Set up Kinesis streaming
- Implement continuous data ingestion

### Tasks
1. **Sensor Data Simulator**
   ```python
   # simulation/sensor_simulator.py
   import numpy as np
   import pandas as pd
   from datetime import datetime, timedelta
   import json
   import boto3
   import time
   from typing import Dict, List
   
   class SensorDataSimulator:
       def __init__(self, kinesis_stream_name: str):
           self.kinesis = boto3.client('kinesis')
           self.stream_name = kinesis_stream_name
           
           # Base sensor values (typical operational ranges)
           self.baseline_values = {
               's2': 641.82,   # T24
               's3': 1589.70,  # T30
               's4': 1400.60,  # T50
               's7': 2.84,     # P30
               's8': 9061.43,  # Nf
               's9': 8444.26,  # Nc
               's11': 2.84,    # Ps30
               's12': 1.30,    # phi
               's13': 9061.43, # NRf
               's14': 8444.26, # NRc
               's15': 1.61,    # BPR
               's17': 1.30,    # htBleed
               's20': 1.30,    # W31
               's21': 1.30,    # W32
           }
           
           # Degradation patterns
           self.degradation_patterns = {
               's3': 0.02,    # T30 increases with HPC degradation
               's7': 0.01,    # P30 increases
               's11': 0.01,   # Ps30 increases
               's9': -0.5,    # Nc decreases (efficiency loss)
               's14': -0.5,   # NRc decreases
           }
           
       def generate_engine_data(self, engine_id: str, cycle: int, 
                              health_index: float = 1.0) -> Dict:
           """Generate sensor data for one engine cycle"""
           
           # Base timestamp
           timestamp = datetime.utcnow()
           
           # Calculate degradation factor (0 = completely degraded, 1 = healthy)
           degradation_factor = health_index
           
           sensor_data = {}
           
           for sensor, baseline in self.baseline_values.items():
               # Add random noise
               noise = np.random.normal(0, baseline * 0.01)  # 1% noise
               
               # Apply degradation if applicable
               if sensor in self.degradation_patterns:
                   degradation = self.degradation_patterns[sensor] * (1 - degradation_factor) * cycle
                   value = baseline + degradation + noise
               else:
                   value = baseline + noise
               
               sensor_data[sensor] = round(value, 2)
           
           # Operational settings (typically constant for a flight)
           operational_data = {
               'altitude': 0.0,      # Sea level for FD001
               'mach_number': 0.25,  # Typical taxi/idle
               'tra': 20.0,          # Throttle position
           }
           
           return {
               'engine_id': engine_id,
               'cycle': cycle,
               'timestamp': timestamp.isoformat(),
               **operational_data,
               **sensor_data,
               'health_index': health_index
           }
       
       def simulate_engine_lifecycle(self, engine_id: str, 
                                   total_cycles: int = 200,
                                   failure_cycle: int = None):
           """Simulate complete engine lifecycle"""
           
           if failure_cycle is None:
               failure_cycle = total_cycles
           
           for cycle in range(1, total_cycles + 1):
               # Calculate health degradation
               health_index = max(0, 1 - (cycle / failure_cycle))
               
               # Generate data
               data = self.generate_engine_data(engine_id, cycle, health_index)
               
               # Send to Kinesis
               self.send_to_kinesis(data)
               
               # Wait between cycles (simulate time passage)
               time.sleep(0.1)  # 10 cycles per second
               
               if cycle % 50 == 0:
                   print(f"Engine {engine_id}: Cycle {cycle}, Health: {health_index:.2f}")
       
       def send_to_kinesis(self, data: Dict):
           """Send sensor data to Kinesis stream"""
           try:
               response = self.kinesis.put_record(
                   StreamName=self.stream_name,
                   Data=json.dumps(data),
                   PartitionKey=data['engine_id']
               )
               return response
           except Exception as e:
               print(f"Error sending to Kinesis: {e}")
       
       def simulate_fleet(self, num_engines: int = 10, 
                         cycles_per_engine: int = 200):
           """Simulate entire fleet of engines"""
           import threading
           
           threads = []
           
           for i in range(1, num_engines + 1):
               engine_id = f"ENGINE_{i:03d}"
               failure_cycle = np.random.randint(150, 250)  # Random failure times
               
               thread = threading.Thread(
                   target=self.simulate_engine_lifecycle,
                   args=(engine_id, cycles_per_engine, failure_cycle)
               )
               threads.append(thread)
               thread.start()
               
               # Stagger engine starts
               time.sleep(1)
           
           # Wait for all engines to complete
           for thread in threads:
               thread.join()
   ```

2. **Kinesis to Supabase Pipeline**
   ```python
   # streaming/kinesis_processor.py
   import json
   import boto3
   from supabase import create_client
   import logging
   
   class KinesisProcessor:
       def __init__(self, supabase_url: str, supabase_key: str):
           self.supabase = create_client(supabase_url, supabase_key)
           self.logger = logging.getLogger(__name__)
           
       def process_records(self, records: List[Dict]):
           """Process Kinesis records and insert into Supabase"""
           
           batch_data = []
           
           for record in records:
               try:
                   # Decode Kinesis record
                   data = json.loads(record['data'])
                   
                   # Get engine UUID from unit number
                   engine_result = self.supabase.table('engines').select('id').eq(
                       'unit_number', data['engine_id']
                   ).execute()
                   
                   if engine_result.data:
                       engine_id = engine_result.data[0]['id']
                       
                       # Prepare record for insertion
                       sensor_record = {
                           'engine_id': engine_id,
                           'cycle': data['cycle'],
                           'timestamp': data['timestamp'],
                           'altitude': data['altitude'],
                           'mach_number': data['mach_number'],
                           'tra': data['tra'],
                       }
                       
                       # Add sensor readings
                       for sensor in ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 
                                     's12', 's13', 's14', 's15', 's17', 's20', 's21']:
                           sensor_record[sensor] = data[sensor]
                       
                       batch_data.append(sensor_record)
               
               except Exception as e:
                   self.logger.error(f"Error processing record: {e}")
                   continue
           
           # Batch insert to Supabase
           if batch_data:
               try:
                   result = self.supabase.table('sensor_data').insert(batch_data).execute()
                   self.logger.info(f"Inserted {len(batch_data)} records")
               except Exception as e:
                   self.logger.error(f"Error inserting batch: {e}")
   
   # AWS Lambda function for Kinesis processing
   def lambda_handler(event, context):
       processor = KinesisProcessor(
           supabase_url=os.environ['SUPABASE_URL'],
           supabase_key=os.environ['SUPABASE_SERVICE_KEY']
       )
       
       records = event['Records']
       processor.process_records(records)
       
       return {
           'statusCode': 200,
           'body': json.dumps(f'Processed {len(records)} records')
       }
   ```

3. **Streaming Infrastructure**
   ```python
   # infrastructure/streaming.py
   import boto3
   
   def setup_kinesis_infrastructure():
       """Set up Kinesis streams and associated resources"""
       
       kinesis = boto3.client('kinesis')
       firehose = boto3.client('firehose')
       
       # Create Kinesis Data Stream
       stream_name = 'sensor-data-stream'
       
       try:
           response = kinesis.create_stream(
               StreamName=stream_name,
               ShardCount=2,  # 2 shards for parallel processing
               StreamModeDetails={
                   'StreamMode': 'PROVISIONED'
               }
           )
           print(f"Created Kinesis stream: {stream_name}")
       except kinesis.exceptions.ResourceInUseException:
           print(f"Stream {stream_name} already exists")
       
       # Create Kinesis Data Firehose for S3 archival
       firehose_name = 'sensor-data-firehose'
       
       try:
           response = firehose.create_delivery_stream(
               DeliveryStreamName=firehose_name,
               DeliveryStreamType='KinesisStreamAsSource',
               KinesisStreamSourceConfiguration={
                   'KinesisStreamARN': f'arn:aws:kinesis:us-east-1:account:stream/{stream_name}',
                   'RoleARN': 'arn:aws:iam::account:role/firehose-delivery-role'
               },
               ExtendedS3DestinationConfiguration={
                   'RoleARN': 'arn:aws:iam::account:role/firehose-delivery-role',
                   'BucketARN': 'arn:aws:s3:::predictive-maintenance-data',
                   'Prefix': 'streaming/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/',
                   'ErrorOutputPrefix': 'errors/',
                   'BufferingHints': {
                       'SizeInMBs': 64,
                       'IntervalInSeconds': 300
                   },
                   'CompressionFormat': 'GZIP',
                   'DataFormatConversionConfiguration': {
                       'Enabled': True,
                       'OutputFormatConfiguration': {
                           'Serializer': {
                               'ParquetSerDe': {}
                           }
                       }
                   }
               }
           )
           print(f"Created Firehose delivery stream: {firehose_name}")
       except Exception as e:
           print(f"Error creating Firehose: {e}")
   ```

### Deliverables
- Real-time sensor data simulator
- Kinesis streaming pipeline
- Lambda function for processing
- S3 archival system

### Testing Checklist
- [ ] Simulator generates realistic data
- [ ] Kinesis stream receives data
- [ ] Lambda processes records
- [ ] Data appears in Supabase

---

## Phase 4: Data Quality and Monitoring
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Implement data quality checks
- Set up anomaly detection
- Create monitoring dashboards

### Tasks
1. **Data Quality Framework**
   ```python
   # quality/quality_monitor.py
   import pandas as pd
   from great_expectations import DataContext
   from typing import Dict, List, Tuple
   import numpy as np
   from scipy import stats
   
   class DataQualityMonitor:
       def __init__(self, supabase_client):
           self.supabase = supabase_client
           self.context = DataContext()
           
       def check_data_completeness(self, table_name: str, 
                                 time_window: str = '1 hour') -> Dict:
           """Check for missing data in recent time window"""
           
           query = f"""
           SELECT 
               engine_id,
               COUNT(*) as record_count,
               MIN(timestamp) as earliest,
               MAX(timestamp) as latest
           FROM {table_name}
           WHERE timestamp >= NOW() - INTERVAL '{time_window}'
           GROUP BY engine_id
           """
           
           result = self.supabase.rpc('execute_sql', {'query': query}).execute()
           
           # Analyze results
           expected_records = self._calculate_expected_records(time_window)
           issues = []
           
           for row in result.data:
               if row['record_count'] < expected_records * 0.9:  # 10% tolerance
                   issues.append({
                       'engine_id': row['engine_id'],
                       'expected': expected_records,
                       'actual': row['record_count'],
                       'completeness': row['record_count'] / expected_records
                   })
           
           return {
               'check_type': 'completeness',
               'status': 'pass' if not issues else 'fail',
               'issues': issues,
               'total_engines_checked': len(result.data)
           }
       
       def check_sensor_ranges(self, engine_id: str = None) -> Dict:
           """Check if sensor values are within expected ranges"""
           
           # Define expected ranges based on C-MAPSS data analysis
           sensor_ranges = {
               's2': (630, 660),    # T24
               's3': (1550, 1650),  # T30
               's4': (1350, 1450),  # T50
               's7': (2.0, 4.0),    # P30
               's8': (8000, 10000), # Nf
               's9': (7000, 9500),  # Nc
               's11': (2.0, 4.0),   # Ps30
               's12': (0.5, 2.0),   # phi
               's13': (8000, 10000),# NRf
               's14': (7000, 9500), # NRc
               's15': (1.0, 2.5),   # BPR
               's17': (0.5, 2.0),   # htBleed
               's20': (0.5, 2.0),   # W31
               's21': (0.5, 2.0),   # W32
           }
           
           outliers = {}
           
           for sensor, (min_val, max_val) in sensor_ranges.items():
               query = f"""
               SELECT engine_id, {sensor}, timestamp
               FROM sensor_data
               WHERE {sensor} < {min_val} OR {sensor} > {max_val}
               AND timestamp >= NOW() - INTERVAL '1 day'
               """
               
               if engine_id:
                   query += f" AND engine_id = '{engine_id}'"
               
               result = self.supabase.rpc('execute_sql', {'query': query}).execute()
               
               if result.data:
                   outliers[sensor] = result.data
           
           return {
               'check_type': 'range_validation',
               'status': 'pass' if not outliers else 'fail',
               'outliers': outliers,
               'sensors_checked': list(sensor_ranges.keys())
           }
       
       def detect_sensor_drift(self, engine_id: str, window_hours: int = 24) -> Dict:
           """Detect statistical drift in sensor values"""
           
           # Get baseline statistics (first 100 cycles)
           baseline_query = f"""
           SELECT s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
           FROM sensor_data
           WHERE engine_id = '{engine_id}'
           ORDER BY cycle
           LIMIT 100
           """
           
           baseline_result = self.supabase.rpc('execute_sql', {'query': baseline_query}).execute()
           baseline_df = pd.DataFrame(baseline_result.data)
           
           # Get recent data
           recent_query = f"""
           SELECT s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
           FROM sensor_data
           WHERE engine_id = '{engine_id}'
           AND timestamp >= NOW() - INTERVAL '{window_hours} hours'
           """
           
           recent_result = self.supabase.rpc('execute_sql', {'query': recent_query}).execute()
           recent_df = pd.DataFrame(recent_result.data)
           
           drift_results = {}
           
           for sensor in baseline_df.columns:
               # Kolmogorov-Smirnov test for distribution drift
               ks_stat, p_value = stats.ks_2samp(
                   baseline_df[sensor].dropna(),
                   recent_df[sensor].dropna()
               )
               
               drift_results[sensor] = {
                   'ks_statistic': ks_stat,
                   'p_value': p_value,
                   'drift_detected': p_value < 0.05,
                   'baseline_mean': baseline_df[sensor].mean(),
                   'recent_mean': recent_df[sensor].mean(),
                   'mean_shift': recent_df[sensor].mean() - baseline_df[sensor].mean()
               }
           
           return {
               'check_type': 'drift_detection',
               'engine_id': engine_id,
               'results': drift_results,
               'summary': {
                   'sensors_with_drift': sum(1 for r in drift_results.values() if r['drift_detected']),
                   'total_sensors': len(drift_results)
               }
           }
       
       def log_quality_metrics(self, check_results: Dict):
           """Log quality check results to database"""
           
           metrics_data = {
               'table_name': 'sensor_data',
               'metric_name': check_results['check_type'],
               'metric_value': 1.0 if check_results['status'] == 'pass' else 0.0,
               'threshold_min': 0.8,
               'threshold_max': 1.0,
               'status': check_results['status'],
               'check_timestamp': datetime.utcnow().isoformat()
           }
           
           self.supabase.table('data_quality_metrics').insert(metrics_data).execute()
   ```

2. **Anomaly Detection**
   ```python
   # quality/anomaly_detector.py
   from sklearn.ensemble import IsolationForest
   from sklearn.preprocessing import StandardScaler
   import numpy as np
   
   class AnomalyDetector:
       def __init__(self, contamination: float = 0.1):
           self.contamination = contamination
           self.scaler = StandardScaler()
           self.detector = IsolationForest(
               contamination=contamination,
               random_state=42
           )
           self.is_fitted = False
       
       def fit(self, normal_data: pd.DataFrame):
           """Fit anomaly detector on normal operating data"""
           
           # Select sensor columns
           sensor_cols = [col for col in normal_data.columns if col.startswith('s')]
           X = normal_data[sensor_cols].dropna()
           
           # Scale features
           X_scaled = self.scaler.fit_transform(X)
           
           # Fit detector
           self.detector.fit(X_scaled)
           self.is_fitted = True
           
       def detect_anomalies(self, data: pd.DataFrame) -> np.ndarray:
           """Detect anomalies in new data"""
           
           if not self.is_fitted:
               raise ValueError("Detector must be fitted before use")
           
           sensor_cols = [col for col in data.columns if col.startswith('s')]
           X = data[sensor_cols].dropna()
           
           # Scale features
           X_scaled = self.scaler.transform(X)
           
           # Predict anomalies (-1 = anomaly, 1 = normal)
           predictions = self.detector.predict(X_scaled)
           scores = self.detector.decision_function(X_scaled)
           
           return predictions, scores
       
       def real_time_monitoring(self, engine_id: str, supabase_client):
           """Monitor real-time data for anomalies"""
           
           # Get recent data
           query = f"""
           SELECT s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21, timestamp
           FROM sensor_data
           WHERE engine_id = '{engine_id}'
           ORDER BY timestamp DESC
           LIMIT 10
           """
           
           result = supabase_client.rpc('execute_sql', {'query': query}).execute()
           
           if not result.data:
               return None
           
           recent_df = pd.DataFrame(result.data)
           predictions, scores = self.detect_anomalies(recent_df)
           
           # Alert on anomalies
           anomaly_indices = np.where(predictions == -1)[0]
           
           if len(anomaly_indices) > 0:
               alerts = []
               for idx in anomaly_indices:
                   alerts.append({
                       'engine_id': engine_id,
                       'timestamp': recent_df.iloc[idx]['timestamp'],
                       'anomaly_score': scores[idx],
                       'sensor_values': recent_df.iloc[idx].to_dict()
                   })
               
               return alerts
           
           return None
   ```

3. **Monitoring Dashboard Data**
   ```python
   # monitoring/dashboard_data.py
   from datetime import datetime, timedelta
   
   class MonitoringDataProvider:
       def __init__(self, supabase_client):
           self.supabase = supabase_client
       
       def get_data_quality_summary(self, hours: int = 24) -> Dict:
           """Get data quality summary for dashboard"""
           
           query = f"""
           SELECT 
               metric_name,
               AVG(metric_value) as avg_quality,
               COUNT(*) as check_count,
               SUM(CASE WHEN status = 'pass' THEN 1 ELSE 0 END) as passed_checks
           FROM data_quality_metrics
           WHERE check_timestamp >= NOW() - INTERVAL '{hours} hours'
           GROUP BY metric_name
           """
           
           result = self.supabase.rpc('execute_sql', {'query': query}).execute()
           
           summary = {
               'overall_quality': 0,
               'checks_passed': 0,
               'total_checks': 0,
               'by_metric': {}
           }
           
           for row in result.data:
               metric_name = row['metric_name']
               avg_quality = row['avg_quality']
               check_count = row['check_count']
               passed_checks = row['passed_checks']
               
               summary['by_metric'][metric_name] = {
                   'avg_quality': avg_quality,
                   'success_rate': passed_checks / check_count if check_count > 0 else 0,
                   'check_count': check_count
               }
               
               summary['total_checks'] += check_count
               summary['checks_passed'] += passed_checks
           
           if summary['total_checks'] > 0:
               summary['overall_quality'] = summary['checks_passed'] / summary['total_checks']
           
           return summary
       
       def get_engine_health_overview(self) -> List[Dict]:
           """Get current health status of all engines"""
           
           query = """
           SELECT 
               e.unit_number,
               e.status,
               COUNT(sd.id) as recent_data_points,
               MAX(sd.timestamp) as last_data_timestamp,
               AVG(p.predicted_rul) as avg_predicted_rul
           FROM engines e
           LEFT JOIN sensor_data sd ON e.id = sd.engine_id 
               AND sd.timestamp >= NOW() - INTERVAL '1 hour'
           LEFT JOIN rul_predictions p ON e.id = p.engine_id 
               AND p.prediction_timestamp >= NOW() - INTERVAL '1 day'
           GROUP BY e.id, e.unit_number, e.status
           ORDER BY e.unit_number
           """
           
           result = self.supabase.rpc('execute_sql', {'query': query}).execute()
           
           return result.data
   ```

### Deliverables
- Data quality monitoring system
- Anomaly detection algorithms
- Quality metrics dashboard
- Automated alerting

### Testing Checklist
- [ ] Quality checks execute successfully
- [ ] Anomalies detected correctly
- [ ] Metrics logged to database
- [ ] Alerts trigger on issues

---

## Phase 5: Data Orchestration and Automation
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Set up Apache Airflow
- Create automated workflows
- Implement error handling

### Tasks
1. **Airflow Setup on AWS MWAA**
   ```python
   # airflow/dags/data_pipeline_dag.py
   from datetime import datetime, timedelta
   from airflow import DAG
   from airflow.operators.python_operator import PythonOperator
   from airflow.operators.bash_operator import BashOperator
   from airflow.sensors.s3_key_sensor import S3KeySensor
   from airflow.providers.postgres.operators.postgres import PostgresOperator
   
   default_args = {
       'owner': 'data-engineering',
       'depends_on_past': False,
       'start_date': datetime(2024, 1, 1),
       'email_on_failure': True,
       'email_on_retry': False,
       'retries': 2,
       'retry_delay': timedelta(minutes=5),
       'email': ['data-team@company.com']
   }
   
   dag = DAG(
       'predictive_maintenance_data_pipeline',
       default_args=default_args,
       description='Complete data pipeline for predictive maintenance',
       schedule_interval=timedelta(hours=1),  # Run every hour
       catchup=False,
       max_active_runs=1
   )
   
   def run_data_quality_checks():
       """Run comprehensive data quality checks"""
       from quality.quality_monitor import DataQualityMonitor
       from supabase import create_client
       
       supabase = create_client(
           Variable.get("SUPABASE_URL"),
           Variable.get("SUPABASE_SERVICE_KEY")
       )
       
       monitor = DataQualityMonitor(supabase)
       
       # Run all quality checks
       checks = [
           monitor.check_data_completeness('sensor_data'),
           monitor.check_sensor_ranges(),
       ]
       
       # Check each engine for drift
       engines_result = supabase.table('engines').select('id, unit_number').execute()
       
       for engine in engines_result.data:
           drift_check = monitor.detect_sensor_drift(engine['id'])
           checks.append(drift_check)
           monitor.log_quality_metrics(drift_check)
       
       # Aggregate results
       failed_checks = [check for check in checks if check['status'] == 'fail']
       
       if failed_checks:
           raise ValueError(f"Data quality checks failed: {len(failed_checks)} issues")
       
       print(f"All data quality checks passed: {len(checks)} checks")
   
   def detect_anomalies():
       """Run anomaly detection on recent data"""
       from quality.anomaly_detector import AnomalyDetector
       
       # Implementation here
       pass
   
   def generate_health_reports():
       """Generate health reports for all engines"""
       from monitoring.dashboard_data import MonitoringDataProvider
       
       # Implementation here
       pass
   
   def trigger_model_retraining():
       """Trigger ML model retraining if needed"""
       # Check data drift levels
       # Trigger SageMaker pipeline if drift is significant
       pass
   
   # Tasks
   quality_check_task = PythonOperator(
       task_id='run_quality_checks',
       python_callable=run_data_quality_checks,
       dag=dag
   )
   
   anomaly_detection_task = PythonOperator(
       task_id='detect_anomalies',
       python_callable=detect_anomalies,
       dag=dag
   )
   
   health_report_task = PythonOperator(
       task_id='generate_health_reports',
       python_callable=generate_health_reports,
       dag=dag
   )
   
   model_check_task = PythonOperator(
       task_id='check_model_retraining',
       python_callable=trigger_model_retraining,
       dag=dag
   )
   
   # Dependencies
   quality_check_task >> [anomaly_detection_task, health_report_task] >> model_check_task
   ```

2. **Error Handling and Recovery**
   ```python
   # utils/error_handling.py
   import logging
   from functools import wraps
   from typing import Callable, Any
   import time
   
   def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
       """Decorator for retrying failed operations"""
       def decorator(func: Callable) -> Callable:
           @wraps(func)
           def wrapper(*args, **kwargs) -> Any:
               last_exception = None
               
               for attempt in range(max_retries + 1):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       last_exception = e
                       if attempt < max_retries:
                           logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                           time.sleep(delay * (2 ** attempt))  # Exponential backoff
                       else:
                           logging.error(f"All {max_retries + 1} attempts failed")
               
               raise last_exception
           return wrapper
       return decorator
   
   @retry_on_failure(max_retries=3)
   def safe_supabase_operation(operation: Callable, *args, **kwargs):
       """Safely execute Supabase operations with retry"""
       return operation(*args, **kwargs)
   
   class DataPipelineErrorHandler:
       def __init__(self, notification_sns_topic: str):
           self.sns_topic = notification_sns_topic
           self.sns = boto3.client('sns')
       
       def handle_critical_error(self, error: Exception, context: str):
           """Handle critical errors that require immediate attention"""
           
           message = f"""
           Critical Error in Data Pipeline
           
           Context: {context}
           Error: {str(error)}
           Timestamp: {datetime.utcnow().isoformat()}
           
           Immediate action required.
           """
           
           self.sns.publish(
               TopicArn=self.sns_topic,
               Subject='CRITICAL: Data Pipeline Error',
               Message=message
           )
           
           # Log to CloudWatch
           logging.critical(f"Critical error in {context}: {error}")
       
       def handle_data_quality_failure(self, quality_results: Dict):
           """Handle data quality check failures"""
           
           if quality_results['status'] == 'fail':
               self.handle_critical_error(
                   Exception("Data quality checks failed"),
                   f"Quality check: {quality_results['check_type']}"
               )
   ```

### Deliverables
- Airflow DAGs for automation
- Error handling framework
- Automated quality monitoring
- Pipeline orchestration

### Testing Checklist
- [ ] Airflow DAGs execute successfully
- [ ] Error handling works correctly
- [ ] Automated alerts trigger
- [ ] Pipeline monitoring active

---

## Success Metrics

### Data Quality
- Data completeness: > 98%
- Data accuracy: > 99%
- Schema compliance: 100%
- Anomaly detection accuracy: > 90%

### Pipeline Performance
- Data ingestion latency: < 5 minutes
- Quality check completion: < 10 minutes
- Pipeline availability: > 99.5%
- Error recovery time: < 15 minutes

### Operational Excellence
- Automated monitoring: 100%
- Alert response time: < 5 minutes
- Data lineage tracking: 100%

---

## Risk Mitigation

### Technical Risks
1. **Data Loss**
   - Mitigation: Multi-layer backups, replication

2. **Quality Degradation**
   - Mitigation: Real-time monitoring, automated alerts

3. **Pipeline Failures**
   - Mitigation: Retry logic, error handling, monitoring

### Operational Risks
1. **Storage Costs**
   - Mitigation: Data lifecycle policies, compression

2. **Performance Issues**
   - Mitigation: Monitoring, auto-scaling, optimization

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Data Architecture | 3-4 days | Supabase setup |
| Phase 2: C-MAPSS Ingestion | 3-4 days | Phase 1 |
| Phase 3: Real-time Simulation | 4-5 days | Phase 1, AWS setup |
| Phase 4: Quality Monitoring | 3-4 days | Phase 2-3 |
| Phase 5: Orchestration | 3-4 days | All phases |

**Total Duration**: 16-21 days (3-4 weeks)

---

## Next Steps
1. Set up Supabase database schema
2. Configure AWS Kinesis streams
3. Begin C-MAPSS data ingestion
4. Implement real-time simulation
5. Set up monitoring and quality checks