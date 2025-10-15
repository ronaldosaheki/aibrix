#!/usr/bin/env python3
"""
Instance Availability Scheduler

This script uses APScheduler to run check_instance_availability.sh every hour
to monitor AWS EC2 instance availability across different regions and instance types.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instance_availability_scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to S3 bucket"""
    try:
        s3_client = boto3.client('s3')
        
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        
        s3_client.upload_file(file_path, bucket_name, s3_key)
        
        logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        return True
        
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        return False
    except ClientError as e:
        logger.error(f"AWS S3 error uploading {file_path}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error uploading {file_path} to S3: {str(e)}")
        return False

def run_instance_check():
    """Run the AWS EC2 instance availability check inline"""
    try:
        logger.info("Starting instance availability check...")
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_file = f"instance_availability_{timestamp}.txt"
        
        # Define regions and instance types
        regions = [
            "us-east-1", "us-east-2", "us-west-1", "us-west-2", 
            "eu-north-1", "eu-west-2", "ap-northeast-1", 
            "ap-south-1", "ap-southeast-2"
            #"ap-sortheast-3" # TODO: Add this region when it is available
        ]
        instance_types = ["p5.48xlarge", "p5e.48xlarge"]
        
        # Open output file for writing
        with open(output_file, 'w') as f:
            for region in regions:
                for instance_type in instance_types:
                    logger.info(f"Checking {region} - {instance_type}")
                    f.write(f"{region}\n")
                    f.write(f"{instance_type}\n")
                    
                    # Run AWS CLI command
                    cmd = [
                        "aws", "ec2", "describe-capacity-block-offerings",
                        "--region", region,
                        "--capacity-duration-hours", "24",
                        "--instance-type", instance_type,
                        "--instance-count", "1"
                    ]
                    
                    # Use default environment (no need to set AWS_REGION)
                    env = os.environ.copy()
                    
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,  # 1 minute timeout per command
                            env=env
                        )
                        
                        if result.returncode == 0:
                            f.write(result.stdout)
                            logger.info(f"Successfully checked {region} - {instance_type}")
                        else:
                            error_msg = f"Error checking {region} - {instance_type}: {result.stderr}"
                            logger.error(error_msg)
                            f.write(f"ERROR: {error_msg}\n")
                            
                    except subprocess.TimeoutExpired:
                        error_msg = f"Timeout checking {region} - {instance_type}"
                        logger.error(error_msg)
                        f.write(f"TIMEOUT: {error_msg}\n")
                    except Exception as e:
                        error_msg = f"Exception checking {region} - {instance_type}: {str(e)}"
                        logger.error(error_msg)
                        f.write(f"EXCEPTION: {error_msg}\n")
        
        logger.info(f"Instance availability check completed successfully")
        logger.info(f"Output saved to: {output_file}")
        
        # Upload to S3 (get bucket and prefix from environment variables)
        bucket_name = os.getenv('S3_BUCKET_NAME', 'default')
        s3_prefix = os.getenv('S3_PREFIX', 'default')
        s3_key = f"{s3_prefix}/{output_file}"
        
        upload_success = upload_to_s3(output_file, bucket_name, s3_key)
        if upload_success:
            logger.info(f"File uploaded to S3: s3://{bucket_name}/{s3_key}")
        else:
            logger.warning(f"Failed to upload {output_file} to S3, but local file was created")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during instance check: {str(e)}")
        return False

def main():
    """Main function to set up and start the scheduler"""
    logger.info("Starting Instance Availability Scheduler")
    logger.info("Script will run every 30 minutes")
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Add job to run every 30 minutes
    scheduler.add_job(
        func=run_instance_check,
        trigger=CronTrigger(minute='*/30'),  # Run every 30 minutes
        id='instance_availability_check',
        name='Instance Availability Check',
        replace_existing=True
    )
    
    # Run an initial check
    logger.info("Running initial instance availability check...")
    run_instance_check()
    
    try:
        logger.info("Scheduler started. Press Ctrl+C to exit.")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
        scheduler.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
