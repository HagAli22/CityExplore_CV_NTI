"""
Dataset downloader and initial setup for Egyptian Landmarks project
"""

import os
import subprocess
import logging
from roboflow import Roboflow
from config import DATASET_CONFIG, PATHS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.api_key = DATASET_CONFIG['roboflow_api_key']
        self.workspace = DATASET_CONFIG['workspace']
        self.project_name = DATASET_CONFIG['project']
        self.version = DATASET_CONFIG['version']
        self.format = DATASET_CONFIG['format']
    
    def install_dependencies(self):
        """Install required dependencies"""
        try:
            logger.info("Installing roboflow...")
            subprocess.run(['pip', 'install', '-q', 'roboflow'], check=True)
            logger.info("Roboflow installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def download_dataset(self):
        """Download dataset from Roboflow"""
        try:
            logger.info("Initializing Roboflow connection...")
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project_name)
            version = project.version(self.version)
            
            logger.info(f"Downloading dataset in {self.format} format...")
            dataset = version.download(self.format)
            
            logger.info("Dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            PATHS['dataset_processed'],
            PATHS['cropped_data'],
            PATHS['checkpoints']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

def main():
    """Main function to download and setup dataset"""
    downloader = DatasetDownloader()
    
    # Install dependencies
    if not downloader.install_dependencies():
        return False
    
    # Setup directories
    downloader.setup_directories()
    
    # Download dataset
    if not downloader.download_dataset():
        return False
    
    logger.info("Dataset setup completed successfully!")
    return True

if __name__ == "__main__":
    main()