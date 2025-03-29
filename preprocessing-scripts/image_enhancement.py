import os
import cv2
import numpy as np
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image, ImageEnhance

# Define Paths
# data_dir = "Dataset"
processed_dir = "processed_images_main"
preprocessed_dir = "preprocessed_images_main"

# Delete and recreate preprocessed directory
if os.path.exists(preprocessed_dir):
    shutil.rmtree(preprocessed_dir)
os.makedirs(preprocessed_dir)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    image = enhance_contrast(image)
    image = sharpen_image(image)
    image = remove_noise(image)
    thresholded = adaptive_threshold(image)

    return image, thresholded

def save_image(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def process_and_save_images():
    for split in ["train", "test", "validation"]:
        for label in ["0", "1"]:
            src_folder = os.path.join(processed_dir, split, label)
            dest_folder = os.path.join(preprocessed_dir, split, label)
            os.makedirs(dest_folder, exist_ok=True)

            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)
                processed_image, thresholded = preprocess_image(img_path)

                save_image(processed_image, os.path.join(dest_folder, img_name))
                save_image(thresholded, os.path.join(dest_folder, f"thresh_{img_name}"))

process_and_save_images()

print("Preprocessing Complete. Images stored in:", preprocessed_dir)
