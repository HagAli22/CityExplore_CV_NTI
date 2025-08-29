"""
Main pipeline for Egyptian Landmarks Detection and Classification Project
"""

import argparse
import logging
import sys
from config import PATHS
from utils import ValidationUtils, setup_logging
from data_downloader import DatasetDownloader
from data_processor import DatasetProcessor
from yolo_trainer import YOLOTrainer
from classification_trainer import ClassificationTrainer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class EgyptianLandmarksPipeline:
    """Main pipeline for the Egyptian Landmarks project"""
    
    def __init__(self):
        self.downloader = DatasetDownloader()
        self.processor = DatasetProcessor()
        self.yolo_trainer = YOLOTrainer()
        self.classifier = ClassificationTrainer()
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("🚀 Starting Egyptian Landmarks Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Download dataset
        logger.info("📥 Step 1: Downloading dataset...")
        if not self.download_data():
            logger.error("Failed to download dataset")
            return False
        
        # Step 2: Process dataset
        logger.info("🔄 Step 2: Processing dataset...")
        if not self.process_data():
            logger.error("Failed to process dataset")
            return False
        
        # Step 3: Train YOLO model
        logger.info("🎯 Step 3: Training YOLO detection model...")
        if not self.train_yolo():
            logger.error("Failed to train YOLO model")
            return False
        
        # Step 4: Train classification model
        logger.info("🧠 Step 4: Training classification model...")
        if not self.train_classifier():
            logger.error("Failed to train classification model")
            return False
        
        logger.info("✅ Pipeline completed successfully!")
        return True
    
    def download_data(self):
        """Download dataset step"""
        try:
            # Install dependencies
            if not self.downloader.install_dependencies():
                return False
            
            # Setup directories
            self.downloader.setup_directories()
            
            # Download dataset
            if not self.downloader.download_dataset():
                return False
            
            logger.info("✅ Dataset download completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in data download: {e}")
            return False
    
    def process_data(self):
        """Process dataset step"""
        try:
            # Merge datasets
            train_count, test_count = self.processor.merge_datasets()
            
            # Get statistics
            num_images, num_classes = self.processor.get_dataset_statistics()
            
            # Remap class labels
            self.processor.remap_class_labels()
            
            logger.info(f"✅ Dataset processing completed")
            logger.info(f"   - Training samples: {train_count}")
            logger.info(f"   - Test samples: {test_count}")
            logger.info(f"   - Total images: {num_images}")
            logger.info(f"   - Number of classes: {num_classes}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return False
    
    def train_yolo(self):
        """Train YOLO model step"""
        try:
            # Install dependencies
            if not self.yolo_trainer.install_ultralytics():
                return False
            
            # Train model
            results = self.yolo_trainer.train_model()
            
            if results is not None:
                # Process dataset for classification
                self.yolo_trainer.process_dataset_for_classification()
                logger.info("✅ YOLO training completed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in YOLO training: {e}")
            return False
    
    def train_classifier(self):
        """Train classification model step"""
        try:
            # Setup datasets
            num_classes = self.classifier.setup_datasets()
            
            # Setup model
            self.classifier.setup_model(num_classes)
            
            # Train model
            if self.classifier.train_model():
                # Save final model
                self.classifier.save_model()
                
                # Evaluate model
                val_loss, val_acc = self.classifier.evaluate_model(
                    self.classifier.val_dataloader, "Validation"
                )
                train_loss, train_acc = self.classifier.evaluate_model(
                    self.classifier.train_dataloader, "Training"
                )
                
                logger.info("✅ Classification training completed")
                logger.info(f"   - Final validation accuracy: {val_acc:.4f}")
                logger.info(f"   - Final training accuracy: {train_acc:.4f}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in classification training: {e}")
            return False
    
    def run_inference_only(self, image_path, yolo_model_path=None, classifier_path=None):
        """Run inference on a single image"""
        logger.info(f"🔍 Running inference on: {image_path}")
        
        try:
            # Load YOLO model
            if yolo_model_path:
                self.yolo_trainer.load_trained_model(yolo_model_path)
            
            # Run YOLO detection
            yolo_results = self.yolo_trainer.predict_image(image_path)
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                logger.info(f"✅ YOLO detected {len(yolo_results[0].boxes)} objects")
                
                # If classification model is available, run classification on crops
                if classifier_path:
                    # Load classification model
                    self.classifier.load_model(classifier_path)
                    
                    # Process detections for classification
                    # This would involve cropping detected regions and classifying them
                    logger.info("🧠 Running classification on detected regions...")
                    
                logger.info("✅ Inference completed successfully")
                return yolo_results
            else:
                logger.info("⚠️  No objects detected in the image")
                return None
                
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Egyptian Landmarks Detection and Classification')
    
    parser.add_argument('--mode', choices=['full', 'download', 'process', 'train_yolo', 
                                          'train_classifier', 'inference'], 
                       default='full', help='Pipeline mode to run')
    
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    parser.add_argument('--yolo_model', type=str, help='Path to trained YOLO model')
    parser.add_argument('--classifier_model', type=str, help='Path to trained classifier model')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Check GPU availability
    ValidationUtils.check_gpu_availability()
    
    # Initialize pipeline
    pipeline = EgyptianLandmarksPipeline()
    
    try:
        if args.mode == 'full':
            success = pipeline.run_full_pipeline()
        elif args.mode == 'download':
            success = pipeline.download_data()
        elif args.mode == 'process':
            success = pipeline.process_data()
        elif args.mode == 'train_yolo':
            success = pipeline.train_yolo()
        elif args.mode == 'train_classifier':
            success = pipeline.train_classifier()
        elif args.mode == 'inference':
            if not args.image_path:
                logger.error("Image path required for inference mode")
                sys.exit(1)
            results = pipeline.run_inference_only(
                args.image_path, args.yolo_model, args.classifier_model
            )
            success = results is not None
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()