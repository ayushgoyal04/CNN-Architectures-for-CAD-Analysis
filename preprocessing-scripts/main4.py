import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

# Define Paths
data_dir = "Dataset"
processed_dir = "processed_images4"
labeled_dir = os.path.join(data_dir, "Coronary_Artery_labelled")
unlabeled_dir = os.path.join(data_dir, "Coronary_Artery_unlabelled")

# Delete and recreate processed directory
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.makedirs(os.path.join(processed_dir, "train"))

# Augmentation Pipeline for High-Quality Images
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30, fill=(255, 255, 255)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Load EfficientNetB0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

# Function to Ensure Image Quality (Save with High Quality)
def save_high_quality(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    image.save(path, format="JPEG", quality=95)

# Function to Apply Augmentations
def apply_augmentations(image_path, num_augmentations=10):
    image = Image.open(image_path).convert('RGB')
    augmented_images = []
    for _ in range(num_augmentations):
        transformed_image = augmentation(image)
        augmented_images.append(transformed_image)
    return augmented_images

# Function to Get Pseudo-Labels
def get_pseudo_labels(images, model, threshold=0.6):
    pseudo_labels = []
    for img_path in images:
        image = Image.open(img_path).convert('RGB')
        image = transforms.Resize((224, 224))(image)
        image = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            confidence, label = torch.max(probs, dim=1)
            if confidence.item() > threshold:
                pseudo_labels.append((img_path, label.item()))
    return pseudo_labels

# Get Unlabeled Image Paths
unlabeled_images = []
for folder in ["test", "train", "val"]:
    folder_path = os.path.join(unlabeled_dir, folder)
    for img_name in os.listdir(folder_path):
        unlabeled_images.append(os.path.join(folder_path, img_name))

# Pseudo-Labeling
pseudo_labels = get_pseudo_labels(unlabeled_images, model, threshold=0.6)

# Organizing Data into Labeled Folders
for label in [0, 1]:
    os.makedirs(os.path.join(processed_dir, "train", str(label)), exist_ok=True)

# Move Pseudo-Labeled Images to Processed Folder with Augmentations
for img_path, label in pseudo_labels:
    label_dir = os.path.join(processed_dir, "train", str(label))
    os.makedirs(label_dir, exist_ok=True)
    augmented_images = apply_augmentations(img_path, num_augmentations=10)
    for i, aug_img in enumerate(augmented_images):
        aug_img_pil = transforms.ToPILImage()(aug_img)
        save_high_quality(aug_img_pil, os.path.join(label_dir, f"{os.path.basename(img_path).split('.')[0]}_aug{i}.jpg"))

# Load and Augment Labeled Data
for category in ["abnormal", "normal"]:
    src_folder = os.path.join(labeled_dir, category)
    label = 1 if category == "abnormal" else 0
    dest_folder = os.path.join(processed_dir, "train", str(label))
    os.makedirs(dest_folder, exist_ok=True)
    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)
        augmented_images = apply_augmentations(img_path, num_augmentations=10)
        for i, aug_img in enumerate(augmented_images):
            aug_img_pil = transforms.ToPILImage()(aug_img)
            save_high_quality(aug_img_pil, os.path.join(dest_folder, f"{img_name.split('.')[0]}_aug{i}.jpg"))

# Final Summary
print("Processed Images Stored in:", processed_dir)
print("Total Processed Images:", sum(len(files) for _, _, files in os.walk(processed_dir)))
