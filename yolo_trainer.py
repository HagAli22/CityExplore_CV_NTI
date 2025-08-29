"""
YOLO model training and object detection for Egyptian Landmarks
"""

import os
import subprocess
import logging
from ultralytics import YOLO
from PIL import Image
from config import PATHS, YOLO_CONFIG, CLASS_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self):
        self.dataset_path = PATHS['dataset_processed']
        self.model_size = YOLO_CONFIG['model_size']
        self.epochs = YOLO_CONFIG['epochs']
        self.confidence = YOLO_CONFIG['confidence']
        self.device = YOLO_CONFIG['device']
        self.project_name = YOLO_CONFIG['project_name']
        self.run_name = YOLO_CONFIG['run_name']
        self.model = None
    
    def install_ultralytics(self):
        """Install ultralytics package"""
        try:
            logger.info("Installing ultralytics...")
            subprocess.run(['pip', 'install', '-q', 'ultralytics'], check=True)
            logger.info("Ultralytics installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ultralytics: {e}")
            return False
    
    def create_yaml_config(self):
        """Create YAML configuration file for YOLO training"""
        yaml_content = f"""
path: {self.dataset_path}
train: {self.dataset_path}/images/train
val: {self.dataset_path}/images/test

names:
"""
        
        for class_id, class_name in CLASS_NAMES.items():
            yaml_content += f"  {class_id}: {class_name}\n"
        
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"YAML configuration created at: {yaml_path}")
        return yaml_path
    
    def train_model(self):
        """Train YOLO model"""
        try:
            logger.info("Initializing YOLO model...")
            self.model = YOLO(self.model_size)
            
            logger.info("Starting YOLO training...")
            yaml_path = self.create_yaml_config()
            
            results = self.model.train(
                data=yaml_path,
                epochs=self.epochs,
                project=self.project_name,
                name=self.run_name,
                device=self.device
            )
            
            logger.info("YOLO training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train YOLO model: {e}")
            return None
    
    def load_trained_model(self, model_path=None):
        """Load trained YOLO model"""
        if model_path is None:
            model_path = f'{self.project_name}/{self.run_name}/weights/best.pt'
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_image(self, image_path, save_results=True):
        """Make prediction on a single image"""
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return None
        
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.confidence,
                device=self.device,
                save=save_results
            )
            
            logger.info(f"Prediction completed for: {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return None
    
    def crop_and_save_detections(self, images_dir, output_dir):
        """Crop detected objects and save them by class"""
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return False
        
        logger.info(f"Processing images from: {images_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            img_path = os.path.join(images_dir, img_name)
            
            # Run YOLO inference
            results = self.model(img_path)
            
            # Open image
            img = Image.open(img_path).convert("RGB")
            
            # Process each detection
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0].item())
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                
                # Crop the detection
                crop = img.crop((x_min, y_min, x_max, y_max))
                
                # Create class directory
                class_dir = os.path.join(output_dir, str(cls_id))
                os.makedirs(class_dir, exist_ok=True)
                
                # Save cropped image
                crop_name = f"{img_name.replace('.jpg', '')}_{i}.jpg"
                crop.save(os.path.join(class_dir, crop_name))
        
        logger.info(f"✅ Cropping completed: {images_dir} → {output_dir}")
        return True
    
    def process_dataset_for_classification(self):
        """Process entire dataset to create cropped images for classification"""
        train_dir = os.path.join(self.dataset_path, "images", "train")
        test_dir = os.path.join(self.dataset_path, "images", "test")
        
        cropped_base = PATHS['cropped_data']
        
        # Process training data
        self.crop_and_save_detections(
            train_dir, 
            os.path.join(cropped_base, "train")
        )
        
        # Process test data
        self.crop_and_save_detections(
            test_dir,
            os.path.join(cropped_base, "test")
        )
        
        logger.info("Dataset processing for classification completed!")

def main():
    """Main function to train YOLO model"""
    trainer = YOLOTrainer()
    
    # Install dependencies
    if not trainer.install_ultralytics():
        return False
    
    # Train model
    results = trainer.train_model()
    
    if results is not None:
        # Process dataset for classification
        trainer.process_dataset_for_classification()
        logger.info("YOLO training and processing completed successfully!")
        return True
    
    return False

if __name__ == "__main__":
    main()