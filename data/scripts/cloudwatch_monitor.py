"""
CloudWatch Monitoring and Alerting
Sends metrics and creates alarms for data pipeline monitoring
"""

import boto3
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import get_config

# =====================================================
# Logging Setup
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# CloudWatch Monitor Class
# =====================================================

class CloudWatchMonitor:
    """CloudWatch monitoring and alerting for data pipeline"""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize AWS clients
        try:
            self.cloudwatch = boto3.client('cloudwatch')
            self.sns = boto3.client('sns')
            logger.info("AWS CloudWatch client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
        
        # Monitoring configuration
        self.namespace = "PredictiveMaintenance/DataPipeline"
        self.dimensions = [
            {'Name': 'Environment', 'Value': self.config.environment},
            {'Name': 'Component', 'Value': 'DataPipeline'}
        ]
        
        # Metric definitions
        self.metrics = {
            'data_quality': {
                'MetricName': 'DataQualityScore',
                'Unit': 'Percent',
                'Description': 'Overall data quality score'
            },
            'anomaly_detection': {
                'MetricName': 'AnomalyCount',
                'Unit': 'Count',
                'Description': 'Number of anomalies detected'
            },
            'pipeline_success': {
                'MetricName': 'PipelineSuccess',
                'Unit': 'Count',
                'Description': 'Pipeline execution success'
            },
            'processing_time': {
                'MetricName': 'ProcessingTimeMS',
                'Unit': 'Milliseconds',
                'Description': 'Data processing time'
            },
            'records_processed': {
                'MetricName': 'RecordsProcessed',
                'Unit': 'Count',
                'Description': 'Number of records processed'
            },
            'error_rate': {
                'MetricName': 'ErrorRate',
                'Unit': 'Percent',
                'Description': 'Pipeline error rate'
            },
            'memory_usage': {
                'MetricName': 'MemoryUsageMB',
                'Unit': 'Megabytes',
                'Description': 'Memory usage in MB'
            },
            'database_connections': {
                'MetricName': 'DatabaseConnections',
                'Unit': 'Count',
                'Description': 'Active database connections'
            }
        }
        
        # Alarm configurations
        self.alarm_configs = {
            'DataQualityLow': {
                'MetricName': 'DataQualityScore',
                'Threshold': 70,
                'ComparisonOperator': 'LessThanThreshold',
                'AlarmDescription': 'Data quality score is below acceptable threshold',
                'AlarmActions': ['data-quality-alert']
            },
            'HighAnomalyRate': {
                'MetricName': 'AnomalyCount',
                'Threshold': 10,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'High number of anomalies detected',
                'AlarmActions': ['anomaly-alert']
            },
            'PipelineFailure': {
                'MetricName': 'PipelineSuccess',
                'Threshold': 0,
                'ComparisonOperator': 'LessThanOrEqualToThreshold',
                'AlarmDescription': 'Data pipeline execution failed',
                'AlarmActions': ['pipeline-failure-alert']
            },
            'HighErrorRate': {
                'MetricName': 'ErrorRate',
                'Threshold': 10,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Pipeline error rate is too high',
                'AlarmActions': ['error-rate-alert']
            },
            'SlowProcessing': {
                'MetricName': 'ProcessingTimeMS',
                'Threshold': 5000,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Data processing is taking too long',
                'AlarmActions': ['performance-alert']
            },
            'HighMemoryUsage': {
                'MetricName': 'MemoryUsageMB',
                'Threshold': 400,  # 80% of 512MB free tier limit
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Memory usage is approaching free tier limits',
                'AlarmActions': ['resource-alert']
            }
        }
    
    def send_metric(self, metric_name: str, value: float, unit: str = 'Count', 
                   dimensions: Optional[List[Dict]] = None, timestamp: Optional[datetime] = None) -> bool:
        """Send a single metric to CloudWatch"""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Dimensions': dimensions or self.dimensions
            }
            
            if timestamp:
                metric_data['Timestamp'] = timestamp
            
            response = self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
            
            logger.info(f"Sent metric {metric_name}: {value} {unit}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send metric {metric_name}: {e}")
            return False
    
    def send_metrics_batch(self, metrics: List[Dict]) -> int:
        """Send multiple metrics in batch (max 20 per batch)"""
        success_count = 0
        
        # CloudWatch allows max 20 metrics per batch
        for i in range(0, len(metrics), 20):
            batch = metrics[i:i+20]
            
            try:
                # Add default dimensions to each metric
                for metric in batch:
                    if 'Dimensions' not in metric:
                        metric['Dimensions'] = self.dimensions
                
                response = self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=batch
                )
                
                success_count += len(batch)
                logger.info(f"Sent batch of {len(batch)} metrics")
                
            except Exception as e:
                logger.error(f"Failed to send metric batch: {e}")
        
        return success_count
    
    def send_pipeline_metrics(self, pipeline_metrics: Dict) -> bool:
        """Send comprehensive pipeline metrics"""
        try:
            metrics = []
            
            # Data quality metrics
            if 'data_quality_score' in pipeline_metrics:
                metrics.append({
                    'MetricName': 'DataQualityScore',
                    'Value': pipeline_metrics['data_quality_score'],
                    'Unit': 'Percent'
                })
            
            # Processing metrics
            if 'total_records_processed' in pipeline_metrics:
                metrics.append({
                    'MetricName': 'RecordsProcessed',
                    'Value': pipeline_metrics['total_records_processed'],
                    'Unit': 'Count'
                })
            
            if 'processing_time_ms' in pipeline_metrics:
                metrics.append({
                    'MetricName': 'ProcessingTimeMS',
                    'Value': pipeline_metrics['processing_time_ms'],
                    'Unit': 'Milliseconds'
                })
            
            # Error metrics
            if 'error_count' in pipeline_metrics and 'total_records_processed' in pipeline_metrics:
                error_rate = (pipeline_metrics['error_count'] / max(1, pipeline_metrics['total_records_processed'])) * 100
                metrics.append({
                    'MetricName': 'ErrorRate',
                    'Value': error_rate,
                    'Unit': 'Percent'
                })
            
            # Memory usage
            if 'memory_usage_mb' in pipeline_metrics:
                metrics.append({
                    'MetricName': 'MemoryUsageMB',
                    'Value': pipeline_metrics['memory_usage_mb'],
                    'Unit': 'Megabytes'
                })
            
            # Anomaly detection
            if 'anomaly_count' in pipeline_metrics:
                metrics.append({
                    'MetricName': 'AnomalyCount',
                    'Value': pipeline_metrics['anomaly_count'],
                    'Unit': 'Count'
                })
            
            # Pipeline success/failure
            pipeline_success = 1 if pipeline_metrics.get('success', False) else 0
            metrics.append({
                'MetricName': 'PipelineSuccess',
                'Value': pipeline_success,
                'Unit': 'Count'
            })
            
            # Send metrics batch
            success_count = self.send_metrics_batch(metrics)
            return success_count == len(metrics)
            
        except Exception as e:
            logger.error(f"Failed to send pipeline metrics: {e}")
            return False
    
    def create_sns_topics(self) -> Dict[str, str]:
        """Create SNS topics for alerts"""
        topics = {
            'data-quality-alert': 'Data Quality Alerts',
            'anomaly-alert': 'Anomaly Detection Alerts',
            'pipeline-failure-alert': 'Pipeline Failure Alerts',
            'error-rate-alert': 'Error Rate Alerts',
            'performance-alert': 'Performance Alerts',
            'resource-alert': 'Resource Usage Alerts'
        }
        
        topic_arns = {}
        
        for topic_name, description in topics.items():
            try:
                response = self.sns.create_topic(
                    Name=f"predictive-maintenance-{topic_name}",
                    Attributes={
                        'DisplayName': description
                    }
                )
                topic_arns[topic_name] = response['TopicArn']
                logger.info(f"Created SNS topic: {topic_name}")
                
            except Exception as e:
                logger.error(f"Failed to create SNS topic {topic_name}: {e}")
        
        return topic_arns
    
    def create_alarms(self, topic_arns: Dict[str, str]) -> List[str]:
        """Create CloudWatch alarms"""
        created_alarms = []
        
        for alarm_name, config in self.alarm_configs.items():
            try:
                # Get topic ARNs for alarm actions
                alarm_actions = []
                for action in config['AlarmActions']:
                    if action in topic_arns:
                        alarm_actions.append(topic_arns[action])
                
                response = self.cloudwatch.put_metric_alarm(
                    AlarmName=f"PredictiveMaintenance-{alarm_name}",
                    AlarmDescription=config['AlarmDescription'],
                    ActionsEnabled=True,
                    AlarmActions=alarm_actions,
                    MetricName=config['MetricName'],
                    Namespace=self.namespace,
                    Statistic='Average',
                    Dimensions=self.dimensions,
                    Period=300,  # 5 minutes
                    EvaluationPeriods=2,
                    Threshold=config['Threshold'],
                    ComparisonOperator=config['ComparisonOperator'],
                    TreatMissingData='notBreaching'
                )
                
                created_alarms.append(alarm_name)
                logger.info(f"Created alarm: {alarm_name}")
                
            except Exception as e:
                logger.error(f"Failed to create alarm {alarm_name}: {e}")
        
        return created_alarms
    
    def setup_monitoring(self, email_address: Optional[str] = None) -> Dict[str, any]:
        """Set up complete monitoring infrastructure"""
        logger.info("Setting up CloudWatch monitoring infrastructure...")
        
        results = {
            'sns_topics_created': [],
            'alarms_created': [],
            'email_subscribed': False,
            'success': False
        }
        
        try:
            # Create SNS topics
            topic_arns = self.create_sns_topics()
            results['sns_topics_created'] = list(topic_arns.keys())
            
            # Subscribe email to topics if provided
            if email_address:
                for topic_name, topic_arn in topic_arns.items():
                    try:
                        self.sns.subscribe(
                            TopicArn=topic_arn,
                            Protocol='email',
                            Endpoint=email_address
                        )
                        logger.info(f"Subscribed {email_address} to {topic_name}")
                        results['email_subscribed'] = True
                    except Exception as e:
                        logger.warning(f"Failed to subscribe email to {topic_name}: {e}")
            
            # Create CloudWatch alarms
            created_alarms = self.create_alarms(topic_arns)
            results['alarms_created'] = created_alarms
            
            # Create dashboard
            self.create_dashboard()
            
            results['success'] = len(created_alarms) > 0
            
            logger.info(f"Monitoring setup complete: {len(created_alarms)} alarms, {len(topic_arns)} topics")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            results['error'] = str(e)
        
        return results
    
    def create_dashboard(self) -> bool:
        """Create CloudWatch dashboard"""
        try:
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0, "y": 0, "width": 12, "height": 6,
                        "properties": {
                            "metrics": [
                                [self.namespace, "DataQualityScore", "Environment", self.config.environment],
                                [".", "RecordsProcessed", ".", "."],
                                [".", "AnomalyCount", ".", "."]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": "ap-southeast-1",
                            "title": "Data Pipeline Overview"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0, "y": 6, "width": 6, "height": 6,
                        "properties": {
                            "metrics": [
                                [self.namespace, "ErrorRate", "Environment", self.config.environment],
                                [".", "PipelineSuccess", ".", "."]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": "ap-southeast-1",
                            "title": "Pipeline Health"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 6, "y": 6, "width": 6, "height": 6,
                        "properties": {
                            "metrics": [
                                [self.namespace, "ProcessingTimeMS", "Environment", self.config.environment],
                                [".", "MemoryUsageMB", ".", "."]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": "ap-southeast-1",
                            "title": "Performance Metrics"
                        }
                    }
                ]
            }
            
            response = self.cloudwatch.put_dashboard(
                DashboardName="PredictiveMaintenance-DataPipeline",
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info("Created CloudWatch dashboard: PredictiveMaintenance-DataPipeline")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def send_test_metrics(self) -> bool:
        """Send test metrics to verify monitoring setup"""
        logger.info("Sending test metrics...")
        
        test_metrics = {
            'data_quality_score': 85.5,
            'total_records_processed': 1000,
            'processing_time_ms': 2500,
            'error_count': 2,
            'memory_usage_mb': 256,
            'anomaly_count': 3,
            'success': True
        }
        
        return self.send_pipeline_metrics(test_metrics)

# =====================================================
# Utility Functions
# =====================================================

def setup_monitoring_cli():
    """CLI function to setup monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup CloudWatch Monitoring')
    parser.add_argument('--email', help='Email address for alert notifications')
    parser.add_argument('--test', action='store_true', help='Send test metrics')
    
    args = parser.parse_args()
    
    try:
        monitor = CloudWatchMonitor()
        
        # Setup monitoring infrastructure
        results = monitor.setup_monitoring(args.email)
        
        print("\n" + "=" * 60)
        print("CLOUDWATCH MONITORING SETUP RESULTS")
        print("=" * 60)
        print(f"Success: {results['success']}")
        print(f"SNS Topics Created: {len(results['sns_topics_created'])}")
        print(f"Alarms Created: {len(results['alarms_created'])}")
        print(f"Email Subscribed: {results['email_subscribed']}")
        
        if results['sns_topics_created']:
            print("\nSNS Topics:")
            for topic in results['sns_topics_created']:
                print(f"  - {topic}")
        
        if results['alarms_created']:
            print("\nAlarms Created:")
            for alarm in results['alarms_created']:
                print(f"  - {alarm}")
        
        # Send test metrics if requested
        if args.test:
            print("\nSending test metrics...")
            success = monitor.send_test_metrics()
            print(f"Test metrics sent: {success}")
        
        print("\n" + "=" * 60)
        
        if results['success']:
            print("✅ Monitoring setup completed successfully!")
            if args.email:
                print(f"📧 Check your email ({args.email}) to confirm SNS subscriptions")
            print("🔗 View dashboard: https://console.aws.amazon.com/cloudwatch/home?region=ap-southeast-1#dashboards:name=PredictiveMaintenance-DataPipeline")
        else:
            print("❌ Monitoring setup failed")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
        return results['success']
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

# =====================================================
# Integration with Data Pipeline
# =====================================================

def send_pipeline_metrics_to_cloudwatch(pipeline_metrics: Dict):
    """Send pipeline metrics to CloudWatch (for integration)"""
    try:
        monitor = CloudWatchMonitor()
        return monitor.send_pipeline_metrics(pipeline_metrics)
    except Exception as e:
        logger.error(f"Failed to send metrics to CloudWatch: {e}")
        return False

# =====================================================
# Main Function
# =====================================================

def main():
    """Main function for CloudWatch monitoring"""
    setup_monitoring_cli()

if __name__ == "__main__":
    main()