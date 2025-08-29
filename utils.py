"""
Utility functions for Egyptian Landmarks Detection and Classification Project
"""

import os
import shutil
import logging
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataUtils:
    """Data utility functions"""
    
    @staticmethod
    def create_zip_archive(source_path, zip_path):
        """Create a zip archive of a directory"""
        try:
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', source_path)
            logger.info(f"Archive created: {zip_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return False
    
    @staticmethod
    def download_file_colab(file_path):
        """Download file in Google Colab environment"""
        try:
            from google.colab import files
            files.download(file_path)
            logger.info(f"File downloaded: {file_path}")
            return True
        except ImportError:
            logger.warning("Google Colab not available. Skipping download.")
            return False
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False

class VisualizationUtils:
    """Visualization utility functions"""
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, val_accuracies):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_predictions(model, dataloader, device, num_images=8):
        """Display model predictions"""
        model.eval()
        images_shown = 0
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                for i in range(inputs.size(0)):
                    if images_shown >= num_images:
                        break
                    
                    # Denormalize image for display
                    img = inputs[i].cpu()
                    img = img * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                    img = torch.clamp(img, 0, 1)
                    
                    # Plot image
                    axes[images_shown].imshow(img.permute(1, 2, 0))
                    axes[images_shown].set_title(
                        f'True: {CLASS_NAMES.get(labels[i].item(), "Unknown")}\n'
                        f'Pred: {CLASS_NAMES.get(predicted[i].item(), "Unknown")}'
                    )
                    axes[images_shown].axis('off')
                    
                    images_shown += 1
                
                if images_shown >= num_images:
                    break
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_class_distribution(dataset):
        """Plot class distribution in dataset"""
        class_counts = {}
        
        for _, label in dataset:
            class_name = CLASS_NAMES.get(label, f"Class_{label}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        plt.figure(figsize=(12, 8))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return class_counts

class ModelUtils:
    """Model utility functions"""
    
    @staticmethod
    def count_parameters(model):
        """Count the number of trainable parameters in a model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return total_params, trainable_params
    
    @staticmethod
    def predict_single_image(model, image_path, device, transform=None):
        """Make prediction on a single image"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, 1).item()
            confidence = probabilities[predicted_class].item()
        
        class_name = CLASS_NAMES.get(predicted_class, f"Class_{predicted_class}")
        
        result = {
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()
        }
        
        return result
    
    @staticmethod
    def get_model_summary(model, input_size=(3, 224, 224)):
        """Get model summary"""
        try:
            from torchsummary import summary
            summary(model, input_size)
        except ImportError:
            logger.warning("torchsummary not available. Install with: pip install torchsummary")
            
            # Basic model info
            total_params, trainable_params = ModelUtils.count_parameters(model)
            logger.info(f"Model: {model.__class__.__name__}")
            logger.info(f"Input size: {input_size}")

class ValidationUtils:
    """Validation and testing utilities"""
    
    @staticmethod
    def validate_paths(paths_dict):
        """Validate that all required paths exist"""
        missing_paths = []
        
        for name, path in paths_dict.items():
            if not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            logger.error("Missing paths:")
            for path in missing_paths:
                logger.error(f"  - {path}")
            return False
        
        logger.info("All paths validated successfully")
        return True
    
    @staticmethod
    def check_gpu_availability():
        """Check GPU availability and memory"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"GPU available: {device_name}")
            logger.info(f"Number of GPUs: {device_count}")
            logger.info(f"Current GPU: {current_device}")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {memory_reserved:.2f} GB")
            
            return True
        else:
            logger.info("GPU not available. Using CPU.")
            return False

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def main():
    """Test utility functions"""
    logger.info("Testing utility functions...")
    
    # Test GPU availability
    ValidationUtils.check_gpu_availability()
    
    # Test path validation
    test_paths = {
        'current_dir': '.',
        'config_file': 'config.py'
    }
    ValidationUtils.validate_paths(test_paths)
    
    logger.info("Utility functions test completed!")

if __name__ == "__main__":
    main()