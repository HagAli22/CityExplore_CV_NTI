"""
Data processing utilities for Egyptian Landmarks dataset
"""

import os
import glob
import collections
import shutil
import logging
from sklearn.model_selection import train_test_split
from config import PATHS, SELECTED_CLASSES, TRAIN_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self):
        self.egypt_path = PATHS['dataset_raw']
        self.output_path = PATHS['dataset_processed']
        self.selected_classes = SELECTED_CLASSES
    
    def get_top_classes(self, dataset_path, top_n=35):
        """Get the top N classes by frequency in the dataset"""
        logger.info(f"Analyzing dataset to find top {top_n} classes...")
        
        label_files = glob.glob(os.path.join(dataset_path, "**", "labels", "*.txt"), recursive=True)
        
        class_counts = collections.Counter()
        for lf in label_files:
            with open(lf, "r") as f:
                for line in f:
                    if line.strip():
                        cid = int(line.split()[0])
                        class_counts[cid] += 1
        
        top_classes = [cid for cid, count in class_counts.most_common(top_n)]
        logger.info(f"Top {top_n} classes found: {top_classes}")
        
        return top_classes
    
    def collect_class_subset(self, dataset_path, selected_classes):
        """Collect images and labels for selected classes"""
        logger.info(f"Collecting images for {len(selected_classes)} selected classes...")
        
        all_images, all_labels = [], []
        label_files = glob.glob(os.path.join(dataset_path, "**", "labels", "*.txt"), recursive=True)
        
        for lf in label_files:
            with open(lf, "r") as f:
                lines = f.readlines()
                if not lines:
                    continue
                cid = int(lines[0].split()[0])
                if cid in selected_classes:
                    base = os.path.splitext(os.path.basename(lf))[0]
                    
                    img_candidates = glob.glob(os.path.join(dataset_path, "**", "images", base + ".*"), recursive=True)
                    
                    if img_candidates:
                        all_images.append(img_candidates[0])
                        all_labels.append(lf)
        
        logger.info(f"Collected {len(all_images)} images for selected classes")
        return all_images, all_labels
    
    def merge_datasets(self, top_n=35):
        """Merge and organize datasets"""
        logger.info("Starting dataset merge process...")
        
        # Get top classes
        top_classes = self.get_top_classes(self.egypt_path, top_n=top_n)
        
        # Collect images and labels
        imgs, lbls = self.collect_class_subset(self.egypt_path, top_classes)
        
        logger.info(f"📊 Collected {len(imgs)} images total")
        
        # Split train/test
        train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
            imgs, lbls, 
            test_size=TRAIN_CONFIG['test_size'], 
            random_state=TRAIN_CONFIG['random_state'],
            shuffle=TRAIN_CONFIG['shuffle']
        )
        
        # Create directories
        for split in ["train", "test"]:
            os.makedirs(os.path.join(self.output_path, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "labels", split), exist_ok=True)
        
        # Copy train files
        for img, lbl in zip(train_imgs, train_lbls):
            shutil.copy(img, os.path.join(self.output_path, "images", "train", os.path.basename(img)))
            shutil.copy(lbl, os.path.join(self.output_path, "labels", "train", os.path.basename(lbl)))
        
        # Copy test files
        for img, lbl in zip(test_imgs, test_lbls):
            shutil.copy(img, os.path.join(self.output_path, "images", "test", os.path.basename(img)))
            shutil.copy(lbl, os.path.join(self.output_path, "labels", "test", os.path.basename(lbl)))
        
        logger.info(f"✅ Done! {len(train_imgs)} train + {len(test_imgs)} test saved in {self.output_path}")
        
        return len(train_imgs), len(test_imgs)
    
    def get_dataset_statistics(self):
        """Get dataset statistics"""
        dataset_path = self.output_path
        
        # Count images
        img_files = glob.glob(os.path.join(dataset_path, "images", "**", "*.jpg"), recursive=True)
        img_files += glob.glob(os.path.join(dataset_path, "images", "**", "*.png"), recursive=True)
        num_images = len(img_files)
        
        # Count classes
        class_ids = set()
        label_files = glob.glob(os.path.join(dataset_path, "labels", "**", "*.txt"), recursive=True)
        
        for lf in label_files:
            with open(lf, "r") as f:
                for line in f:
                    if line.strip():
                        cid = int(line.split()[0])
                        class_ids.add(cid)
        
        num_classes = len(class_ids)
        
        logger.info(f"Dataset Statistics:")
        logger.info(f"Number of images: {num_images}")
        logger.info(f"Number of classes: {num_classes}")
        
        return num_images, num_classes
    
    def remap_class_labels(self):
        """Remap class labels to sequential indices (0 to n-1)"""
        logger.info("Remapping class labels...")
        
        # Create mapping
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(self.selected_classes)}
        logger.info(f"Class Mapping: {class_mapping}")
        
        # Update label files
        dataset_path = self.output_path
        label_files = glob.glob(os.path.join(dataset_path, "labels", "**", "*.txt"), recursive=True)
        
        for lf in label_files:
            new_lines = []
            with open(lf, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cid = int(parts[0])
                    if old_cid in class_mapping:
                        parts[0] = str(class_mapping[old_cid])
                        new_lines.append(" ".join(parts))
            
            # Overwrite the file
            with open(lf, "w") as f:
                f.write("\n".join(new_lines))
        
        logger.info(f"✅ Done! Classes remapped from 0 → {len(self.selected_classes)-1}")

def main():
    """Main function to process dataset"""
    processor = DatasetProcessor()
    
    # Merge datasets
    train_count, test_count = processor.merge_datasets()
    
    # Get statistics
    num_images, num_classes = processor.get_dataset_statistics()
    
    # Remap class labels
    processor.remap_class_labels()
    
    logger.info("Dataset processing completed successfully!")
    return True

if __name__ == "__main__":
    main()