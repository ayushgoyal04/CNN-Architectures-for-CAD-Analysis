import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
import random

# Define Paths
data_dir = "Dataset"
processed_dir = "processed_images_3"
labeled_dir = os.path.join(data_dir, "Coronary_Artery_labelled")
unlabeled_dir = os.path.join(data_dir, "Coronary_Artery_unlabelled")

# Create processed_images_3 directory
if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.makedirs(processed_dir)

# Augmentation Pipeline with Additional Techniques
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=20, shear=10),
    transforms.ToTensor(),
])

# Load EfficientNetB0
model = models.efficientnet_b0(pretrained=True) #weights=EfficientNet_B0_Weights.DEFAULT
model.eval()

# Function to Apply Advanced Augmentations
def advanced_augment(image_path, num_augmentations=5):
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
        image = augmentation(image).unsqueeze(0)
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

# Pseudo-Labeling with Lower Threshold
pseudo_labels = get_pseudo_labels(unlabeled_images, model, threshold=0.6)

# Move Pseudo-Labeled Images to Processed Folder with Augmentations
for img_path, label in pseudo_labels:
    label_dir = os.path.join(processed_dir, "train", str(label))
    os.makedirs(label_dir, exist_ok=True)
    augmented_images = advanced_augment(img_path, num_augmentations=10)
    for i, aug_img in enumerate(augmented_images):
        aug_img_pil = transforms.ToPILImage()(aug_img)
        aug_img_pil.save(os.path.join(label_dir, f"{os.path.basename(img_path).split('.')[0]}_aug{i}.jpg"))

# Load and Augment Labeled Data
for category in ["abnormal", "normal"]:
    src_folder = os.path.join(labeled_dir, category)
    label = 1 if category == "abnormal" else 0
    dest_folder = os.path.join(processed_dir, "train", str(label))
    os.makedirs(dest_folder, exist_ok=True)
    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)
        augmented_images = advanced_augment(img_path, num_augmentations=10)
        for i, aug_img in enumerate(augmented_images):
            aug_img_pil = transforms.ToPILImage()(aug_img)
            aug_img_pil.save(os.path.join(dest_folder, f"{img_name.split('.')[0]}_aug{i}.jpg"))

# Final Summary
print("Processed Images Stored in:", processed_dir)
print("Total Processed Images:", sum(len(files) for _, _, files in os.walk(processed_dir)))
