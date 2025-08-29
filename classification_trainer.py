"""
CNN Classification model training for Egyptian Landmarks
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from google.colab import files
from config import PATHS, CLASSIFICATION_CONFIG, IMAGENET_MEAN, IMAGENET_STD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationTrainer:
    def __init__(self):
        self.train_data_dir = os.path.join(PATHS['cropped_data'], 'train')
        self.val_data_dir = os.path.join(PATHS['cropped_data'], 'test')
        self.model_path = PATHS['classification_model_path']
        self.checkpoint_dir = PATHS['checkpoints']
        
        # Model configuration
        self.backbone = CLASSIFICATION_CONFIG['model_backbone']
        self.image_size = CLASSIFICATION_CONFIG['image_size']
        self.batch_size = CLASSIFICATION_CONFIG['batch_size']
        self.epochs = CLASSIFICATION_CONFIG['epochs']
        self.learning_rate = CLASSIFICATION_CONFIG['learning_rate']
        self.num_workers = CLASSIFICATION_CONFIG['num_workers']
        
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataset = None
        self.val_dataset = None
        
        logger.info(f"Using device: {self.device}")
    
    def setup_data_transforms(self):
        """Setup data transformations for training and validation"""
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        
        logger.info("Data transforms setup completed")
    
    def setup_datasets(self):
        """Setup datasets and dataloaders"""
        self.setup_data_transforms()
        
        # Create datasets
        self.train_dataset = datasets.ImageFolder(self.train_data_dir, self.train_transforms)
        self.val_dataset = datasets.ImageFolder(self.val_data_dir, self.val_transforms)
        
        # Create dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Number of classes: {len(self.train_dataset.classes)}")
        
        return len(self.train_dataset.classes)
    
    def setup_model(self, num_classes):
        """Setup model architecture"""
        if self.backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"Model setup completed with {num_classes} classes")
        logger.info(f"Model architecture: {self.backbone}")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Create zip and download
        zip_path = f"{checkpoint_path}.zip"
        os.system(f'zip -j "{zip_path}" "{checkpoint_path}"')
        files.download(zip_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.train_dataset)
        return epoch_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_dataset)
        accuracy = correct_predictions / total_predictions
        
        return epoch_loss, accuracy
    
    def train_model(self):
        """Train the classification model"""
        logger.info("Starting classification model training...")
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch()
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 2 == 0 or epoch < self.epochs:
                self.save_checkpoint(epoch)
        
        logger.info("✅ Training finished.")
        return True
    
    def save_model(self):
        """Save the final trained model"""
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"✅ Model saved successfully to {self.model_path}")
    
    def load_model(self, model_path=None, num_classes=35):
        """Load a saved model"""
        if model_path is None:
            model_path = self.model_path
        
        # Setup model architecture
        if self.backbone == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Setup criterion for evaluation
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"✅ Model loaded successfully from {model_path}")
        return True
    
    def evaluate_model(self, dataloader=None, dataset_name="validation"):
        """Evaluate model performance"""
        if dataloader is None:
            dataloader = self.val_dataloader
        
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return None
        
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        dataset_size = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else total_predictions
        epoch_loss = running_loss / dataset_size
        accuracy = correct_predictions / total_predictions
        
        logger.info(f"{dataset_name} Results:")
        logger.info(f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return epoch_loss, accuracy

def main():
    """Main function to train classification model"""
    trainer = ClassificationTrainer()
    
    # Setup datasets
    num_classes = trainer.setup_datasets()
    
    # Setup model
    trainer.setup_model(num_classes)
    
    # Train model
    if trainer.train_model():
        # Save final model
        trainer.save_model()
        
        # Evaluate on validation set
        trainer.evaluate_model(trainer.val_dataloader, "Validation")
        
        # Evaluate on training set
        trainer.evaluate_model(trainer.train_dataloader, "Training")
        
        logger.info("Classification training completed successfully!")
        return True
    
    return False

if __name__ == "__main__":
    main()