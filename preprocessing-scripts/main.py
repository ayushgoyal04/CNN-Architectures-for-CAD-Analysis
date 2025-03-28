import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Define dataset paths
DATASET_PATH = "Dataset"
LABELED_PATH = os.path.join(DATASET_PATH, "Coronary_Artery_labelled")
UNLABELED_PATH = os.path.join(DATASET_PATH, "Coronary_Artery_unlabelled")
PROCESSED_PATH = "processed_images"

# Create processed_images directory
if os.path.exists(PROCESSED_PATH):
    shutil.rmtree(PROCESSED_PATH)
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Function to load and preprocess images
def load_images_from_folder(folder_path, label=None):
    images = []
    labels = []
    filenames = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    img = preprocess_input(img)
                    images.append(img)
                    filenames.append(img_name)
                    if label is None:
                        labels.append(class_folder)  # Labeled data
                    else:
                        labels.append(label)  # Unlabeled data placeholder
    return np.array(images), np.array(labels), filenames

# Load labeled images
labeled_images, labeled_labels, _ = load_images_from_folder(LABELED_PATH)

# Convert labels to binary (0 for normal, 1 for abnormal)
label_map = {"normal": 0, "abnormal": 1}
labeled_labels = np.array([label_map[label] for label in labeled_labels])

# Split labeled data into train, validation, and test sets
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    labeled_images, labeled_labels, test_size=0.3, random_state=42, stratify=labeled_labels)
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Define image augmentation
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load unlabeled images
unlabeled_images, _, filenames = load_images_from_folder(UNLABELED_PATH, label=-1)

# Load EfficientNetB0 for pseudo-labeling
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Generate pseudo-labels
pseudo_labels = model.predict(unlabeled_images)
high_confidence_indices = np.where((pseudo_labels > 0.9) | (pseudo_labels < 0.1))[0]
pseudo_labeled_images = unlabeled_images[high_confidence_indices]
pseudo_labeled_labels = (pseudo_labels[high_confidence_indices] > 0.5).astype(int)

# Merge pseudo-labeled data with training data
final_train_imgs = np.concatenate([train_imgs, pseudo_labeled_images])
final_train_labels = np.concatenate([train_labels, pseudo_labeled_labels.flatten()])


# Shuffle the final dataset
indices = np.arange(final_train_imgs.shape[0])
np.random.shuffle(indices)
final_train_imgs = final_train_imgs[indices]
final_train_labels = final_train_labels[indices]

# Function to save processed images
def save_images(images, labels, folder_name):
    save_path = os.path.join(PROCESSED_PATH, folder_name)
    os.makedirs(save_path, exist_ok=True)

    for i, img in tqdm(enumerate(images), desc=f"Saving {folder_name} images"):
        label_folder = os.path.join(save_path, str(labels[i]))
        os.makedirs(label_folder, exist_ok=True)
        img_path = os.path.join(label_folder, f"image_{i}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Save images in processed_images directory
save_images(final_train_imgs, final_train_labels, "train")
save_images(val_imgs, val_labels, "validation")
save_images(test_imgs, test_labels, "test")

print("Preprocessing complete. Images saved in 'processed_images' folder.")
