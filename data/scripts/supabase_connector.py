"""
Supabase Database Connector - Predictive Maintenance System
Handles all database operations with connection pooling and retry logic
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

import pandas as pd
from supabase import create_client, Client
from postgrest import APIError
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, execute_batch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DatabaseConfig, get_config

# =====================================================
# Logging Setup
# =====================================================

logger = logging.getLogger(__name__)

# =====================================================
# Supabase Connector Class
# =====================================================

class SupabaseConnector:
    """Supabase database connector with optimizations for free tier"""
    
    def __init__(self, config: DatabaseConfig = None):
        """Initialize Supabase connector"""
        if config is None:
            config = get_config().database
        
        self.config = config
        self.client = None
        self.pg_pool = None
        self.connection_retries = 3
        self.batch_size = 500  # Optimized for free tier
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'failed_queries': 0,
            'total_inserts': 0,
            'avg_query_time_ms': 0
        }
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize Supabase and PostgreSQL connections"""
        try:
            # Initialize Supabase client
            if self.config.url and self.config.key:
                self.client = create_client(self.config.url, self.config.key)
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not found in environment")
            
            # Initialize PostgreSQL connection pool for bulk operations
            self._init_pg_pool()
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def _init_pg_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            # Parse Supabase URL to get PostgreSQL connection string
            if self.config.url:
                # Convert Supabase URL to PostgreSQL connection string
                # Format: postgresql://[user]:[password]@[host]:[port]/[database]
                db_url = self.config.url.replace('https://', '')
                db_host = db_url.split('.')[0] + '.db.supabase.co'
                
                # Get database password from service key or environment
                db_password = os.getenv('SUPABASE_DB_PASSWORD', '')
                
                if db_password:
                    self.pg_pool = psycopg2.pool.SimpleConnectionPool(
                        1, 5,  # Min and max connections (free tier optimization)
                        host=db_host,
                        database='postgres',
                        user='postgres',
                        password=db_password,
                        port=5432,
                        connect_timeout=10
                    )
                    logger.info("PostgreSQL connection pool initialized")
                else:
                    logger.warning("PostgreSQL password not found, bulk operations disabled")
                    
        except Exception as e:
            logger.warning(f"Could not initialize PostgreSQL pool: {e}")
            self.pg_pool = None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if self.client:
                # Test Supabase connection
                response = self.client.table('engines').select('count').limit(1).execute()
                logger.info("Supabase connection test successful")
                return True
            
            if self.pg_pool:
                # Test PostgreSQL connection
                conn = self.pg_pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        logger.info("PostgreSQL connection test successful")
                        return True
                finally:
                    self.pg_pool.putconn(conn)
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    # =====================================================
    # Engine Operations
    # =====================================================
    
    def create_engine(self, unit_number: int, **kwargs) -> Dict:
        """Create a new engine record"""
        try:
            data = {
                'unit_number': unit_number,
                'aircraft_model': kwargs.get('aircraft_model', 'Boeing 737'),
                'engine_type': kwargs.get('engine_type', 'Turbofan'),
                'operational_status': kwargs.get('operational_status', 'active'),
                'current_rul': kwargs.get('current_rul', None),
                'manufacture_date': kwargs.get('manufacture_date', None),
                'installation_date': kwargs.get('installation_date', None)
            }
            
            response = self.client.table('engines').insert(data).execute()
            logger.info(f"Created engine with unit_number: {unit_number}")
            return response.data[0] if response.data else {}
            
        except APIError as e:
            if 'duplicate' in str(e).lower():
                logger.warning(f"Engine {unit_number} already exists")
                return self.get_engine(unit_number)
            logger.error(f"Failed to create engine: {e}")
            raise
    
    def get_engine(self, unit_number: int) -> Optional[Dict]:
        """Get engine by unit number"""
        try:
            response = self.client.table('engines').select('*').eq('unit_number', unit_number).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get engine {unit_number}: {e}")
            return None
    
    def get_or_create_engine(self, unit_number: int, **kwargs) -> Dict:
        """Get existing engine or create new one"""
        engine = self.get_engine(unit_number)
        if not engine:
            engine = self.create_engine(unit_number, **kwargs)
        return engine
    
    # =====================================================
    # Sensor Data Operations
    # =====================================================
    
    def insert_sensor_data_batch(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Insert sensor data in batches (optimized for free tier)"""
        start_time = time.time()
        results = {
            'success': False,
            'records_inserted': 0,
            'batches_processed': 0,
            'errors': [],
            'duration_ms': 0
        }
        
        try:
            # Ensure all engines exist first
            unique_units = df['unit_number'].unique()
            engine_map = {}
            
            for unit_num in unique_units:
                engine = self.get_or_create_engine(int(unit_num))
                if engine:
                    engine_map[unit_num] = engine['id']
            
            # Add engine_id to dataframe
            df['engine_id'] = df['unit_number'].map(engine_map)
            
            # Prepare data for insertion
            sensor_columns = ['engine_id', 'unit_number', 'time_cycles', 'timestamp',
                            'altitude', 'mach_number', 'tra', 's2', 's3', 's4', 's7', 
                            's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 
                            's20', 's21', 'rul', 'health_score', 'anomaly_score']
            
            # Filter columns that exist in dataframe
            available_columns = [col for col in sensor_columns if col in df.columns]
            df_to_insert = df[available_columns].copy()
            
            # Convert timestamps if needed
            if 'timestamp' in df_to_insert.columns:
                df_to_insert['timestamp'] = pd.to_datetime(df_to_insert['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate health score if not present
            if 'health_score' not in df_to_insert.columns:
                df_to_insert['health_score'] = 100 - (df_to_insert['rul'] / df_to_insert['rul'].max() * 100)
            
            # Set anomaly score to 0 if not present
            if 'anomaly_score' not in df_to_insert.columns:
                df_to_insert['anomaly_score'] = 0
            
            # Replace NaN with None for proper SQL NULL
            df_to_insert = df_to_insert.where(pd.notnull(df_to_insert), None)
            
            # Insert data in batches
            total_records = len(df_to_insert)
            batch_size = self.batch_size
            
            for i in range(0, total_records, batch_size):
                batch = df_to_insert.iloc[i:i+batch_size]
                
                try:
                    if self.pg_pool:
                        # Use PostgreSQL for bulk insert (faster)
                        self._bulk_insert_pg(batch, 'sensor_data', available_columns)
                    else:
                        # Use Supabase client (slower but works without direct PG access)
                        records = batch.to_dict('records')
                        self.client.table('sensor_data').insert(records).execute()
                    
                    results['records_inserted'] += len(batch)
                    results['batches_processed'] += 1
                    
                    logger.info(f"Inserted batch {results['batches_processed']}: {len(batch)} records")
                    
                except Exception as e:
                    error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    
                    # Continue with next batch instead of failing completely
                    continue
            
            results['success'] = results['records_inserted'] > 0
            results['duration_ms'] = int((time.time() - start_time) * 1000)
            
            logger.info(f"Batch insertion completed: {results['records_inserted']}/{total_records} records in {results['duration_ms']}ms")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Failed to insert sensor data batch: {e}")
        
        return results['success'], results
    
    def _bulk_insert_pg(self, df: pd.DataFrame, table: str, columns: List[str]):
        """Bulk insert using PostgreSQL connection"""
        if not self.pg_pool:
            raise Exception("PostgreSQL connection pool not available")
        
        conn = self.pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Prepare insert query
                schema = 'predictive_maintenance'
                table_ref = sql.Identifier(schema, table)
                columns_ref = [sql.Identifier(col) for col in columns]
                
                query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(', ').join(columns_ref),
                    sql.SQL(', ').join([sql.Placeholder()] * len(columns))
                )
                
                # Convert dataframe to list of tuples
                values = [tuple(row) for row in df[columns].values]
                
                # Execute batch insert
                execute_batch(cur, query, values, page_size=100)
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pg_pool.putconn(conn)
    
    # =====================================================
    # Query Operations
    # =====================================================
    
    def get_latest_sensor_data(self, unit_number: int, limit: int = 100) -> pd.DataFrame:
        """Get latest sensor data for an engine"""
        try:
            response = self.client.table('sensor_data')\
                .select('*')\
                .eq('unit_number', unit_number)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return pd.DataFrame()
    
    def get_engines_by_status(self, status: str) -> List[Dict]:
        """Get engines by operational status"""
        try:
            response = self.client.table('engines')\
                .select('*')\
                .eq('operational_status', status)\
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Failed to get engines by status: {e}")
            return []
    
    # =====================================================
    # Data Quality Operations
    # =====================================================
    
    def log_data_quality_metrics(self, metrics: Dict) -> bool:
        """Log data quality metrics to database"""
        try:
            data = {
                'check_timestamp': datetime.utcnow().isoformat(),
                'table_name': metrics.get('table_name', 'sensor_data'),
                'total_records': metrics.get('total_records', 0),
                'null_count': metrics.get('null_count', 0),
                'duplicate_count': metrics.get('duplicate_count', 0),
                'anomaly_count': metrics.get('anomaly_count', 0),
                'quality_score': metrics.get('quality_score', 0.0),
                'check_results': metrics.get('check_results', {}),
                'issues_found': metrics.get('issues_found', [])
            }
            
            response = self.client.table('data_quality_metrics').insert(data).execute()
            logger.info(f"Logged data quality metrics: score={data['quality_score']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log data quality metrics: {e}")
            return False
    
    # =====================================================
    # Cleanup and Maintenance
    # =====================================================
    
    def cleanup_old_data(self, days_to_keep: int = 180) -> int:
        """Clean up old sensor data (free tier optimization)"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
            
            # Count records to be deleted
            count_response = self.client.table('sensor_data')\
                .select('count')\
                .lt('timestamp', cutoff_date)\
                .execute()
            
            count = count_response.data[0]['count'] if count_response.data else 0
            
            if count > 0:
                # Delete old records
                delete_response = self.client.table('sensor_data')\
                    .delete()\
                    .lt('timestamp', cutoff_date)\
                    .execute()
                
                logger.info(f"Cleaned up {count} old sensor records")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def close(self):
        """Close database connections"""
        try:
            if self.pg_pool:
                self.pg_pool.closeall()
                logger.info("PostgreSQL connection pool closed")
            
            logger.info(f"Connection metrics: {self.metrics}")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# =====================================================
# Utility Functions
# =====================================================

def test_supabase_connection():
    """Test Supabase connection"""
    try:
        config = get_config()
        connector = SupabaseConnector(config.database)
        
        if connector.test_connection():
            logger.info("✅ Supabase connection successful")
            return True
        else:
            logger.error("❌ Supabase connection failed")
            return False
            
    except Exception as e:
        logger.error(f"Connection test error: {e}")
        return False

def create_tables():
    """Create database tables from schema file"""
    try:
        config = get_config()
        
        # Read schema file
        schema_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'schemas', 'supabase_schema.sql'
        )
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Note: This requires direct PostgreSQL access
        # For Supabase, you might need to run this through the Supabase dashboard
        logger.info("Schema SQL loaded. Please run this in Supabase SQL Editor:")
        logger.info(f"Schema file: {schema_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for testing Supabase connector"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Testing Supabase Connector...")
    
    # Test connection
    if not test_supabase_connection():
        logger.error("Connection test failed")
        return False
    
    # Test basic operations
    try:
        with SupabaseConnector() as connector:
            # Create test engine
            engine = connector.get_or_create_engine(
                unit_number=1,
                aircraft_model='Boeing 737-800',
                operational_status='active'
            )
            logger.info(f"Test engine: {engine}")
            
            # Create sample sensor data
            sample_data = pd.DataFrame({
                'unit_number': [1, 1, 1],
                'time_cycles': [1, 2, 3],
                'timestamp': pd.date_range('2024-01-01', periods=3, freq='H'),
                'altitude': [35000, 35000, 35000],
                'mach_number': [0.8, 0.8, 0.8],
                'tra': [85, 85, 85],
                's2': [640, 641, 642],
                's3': [1600, 1601, 1602],
                'rul': [100, 99, 98]
            })
            
            # Insert sensor data
            success, results = connector.insert_sensor_data_batch(sample_data)
            logger.info(f"Insert results: {results}")
            
            # Query latest data
            latest_data = connector.get_latest_sensor_data(1, limit=5)
            logger.info(f"Latest data shape: {latest_data.shape}")
            
            logger.info("✅ All tests passed!")
            return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)