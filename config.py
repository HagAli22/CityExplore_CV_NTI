"""
Configuration file for Egyptian Landmarks Detection and Classification Project
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    'roboflow_api_key': "N79o2Kg5Jx1ItlvgpiNN",
    'workspace': "gp-j4r5m",
    'project': "augmented-iklfg",
    'version': 1,
    'format': "yolov11"
}

# Paths Configuration
PATHS = {
    'dataset_raw': "/content/augmented-1",
    'dataset_processed': "/content/Merged_Top20_YOLO",
    'cropped_data': "egypt_cropped",
    'checkpoints': "checkpoints",
    'yolo_train_results': 'yolo_train',
    'classification_model_path': "/content/egyptian_landmarks_classification_model.pth"
}

# Model Configuration
YOLO_CONFIG = {
    'model_size': 'yolo11n.pt',
    'epochs': 18,
    'confidence': 0.25,
    'device': [-1, -1],  # Use available GPUs
    'project_name': 'yolo_train',
    'run_name': 'log'
}

CLASSIFICATION_CONFIG = {
    'model_backbone': 'resnet18',
    'image_size': 224,
    'batch_size': 32,
    'epochs': 14,
    'learning_rate': 0.001,
    'num_workers': 2
}

# Class Configuration
SELECTED_CLASSES = [19, 53, 24, 36, 5, 44, 41, 2, 82, 72, 34, 32, 0, 80, 62, 70, 56, 27, 69, 1, 29, 16, 66, 14, 71, 60, 23, 46, 15, 42, 13, 55, 64, 78, 38]

CLASS_NAMES = {
    0: "Colossoi of Memnon",
    1: "Sphinx",
    2: "Great Pyramids of Giza",
    3: "Mask of Tutankhamun",
    4: "Bent Pyramid of King Sneferu",
    5: "Pyramid of Djoser",
    6: "Nefertiti",
    7: "Amenhotep III and Tiye",
    8: "bust of Ramesses II",
    9: "Statue of King Zoser",
    10: "King Thutmose III",
    11: "Isis with her child",
    12: "Akhenaten",
    13: "Statue of Tutankhamun",
    14: "Statue of Ankhesenamun",
    15: "Statue of King Ramses II Luxor Temple",
    16: "Standing Statue of King Ramses II",
    17: "Hatshepsut face",
    18: "Statue of King Ramses II Grand Egyptian Museum",
    19: "Amenhotep III",
    20: "Head Statue of Amenhotep iii",
    21: "Colossal Statue of Queen Hatshepsut",
    22: "Statue of Khafre",
    23: "Colossal Statue of King Senwosret IlI",
    24: "Statue of King Sety Il Holding Standards",
    25: "Statue of Amenmhat I",
    26: "Granite Statue of Tutankhamun",
    27: "Seated Statue of Amenhotep III",
    28: "Colossal Statue of Middle Kingdom King",
    29: "Obelsik Tip of Hatshepsut",
    30: "Colossal Statue of Hormoheb",
    31: "Sphinx of Kings Ramesses ll - Merenptah",
    32: "Statue of God Ptah Ramesses ll Goddess Sekhmet",
    33: "Statue of Snefru",
    34: "Menkaure Statue"
}

# Data Augmentation and Preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Configuration
TRAIN_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'shuffle': True,
    'checkpoint_frequency': 2  # Save checkpoint every N epochs
}