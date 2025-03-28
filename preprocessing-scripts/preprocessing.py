"""
Medical Image Preprocessing Pipeline (Without Training)
Author: Ayush  
Date: 2025-03-27

Features:
- Saves preprocessed and augmented images for CNN models
- Ensures same dataset structure for fair benchmarking
- Leakage-proof dataset splitting

Ensure this data directory structure->
project_root/
├── preprocessing.py
├── processed_data/
│   ├── train/
│   │   ├── Normal/
│   │   ├── Abnormal/
│   ├── val/
│   │   ├── Normal/
│   │   ├── Abnormal/
│   ├── test/
│   │   ├── Normal/
│   │   ├── Abnormal/
│   ├── pseudo_labeled/
│   │   ├── Normal/
│   │   ├── Abnormal/
└── raw_data/
    ├── labeled_data/
    │   ├── Normal/
    │   ├── Abnormal/
    ├── unlabeled_data/
        ├── Train/
        ├── Test/
        ├── Validate/

"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2

###############################
# 1. Utility Functions
###############################

def save_image(image, label, save_path, file_name):
    """
    Save an image as a preprocessed file.

    Args:
        image: TensorFlow image tensor.
        label: Class label (int).
        save_path: Destination folder.
        file_name: File name to save.
    """
    os.makedirs(save_path, exist_ok=True)
    img = tf.keras.preprocessing.image.array_to_img(image)
    img.save(os.path.join(save_path, file_name))


###############################
# 2. Data Preprocessing & Storage
###############################

def load_and_preprocess_data(base_path, save_dir, img_size=(224, 224), batch_size=32):
    """
    Load, preprocess, and save images.

    Args:
        base_path: Path to labeled data (Normal/Abnormal subfolders).
        save_dir: Path to save processed images.
        img_size: Image size (height, width).
        batch_size: Batch size.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        base_path,
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42,
        validation_split=0.3,
        subset='training'
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42,
        validation_split=0.3,
        subset='validation'
    )

    val_size = int(0.5 * len(temp_ds))
    val_ds = temp_ds.take(val_size)
    test_ds = temp_ds.skip(val_size)

    # Normalization function
    normalization = layers.Rescaling(1./255)

    # Augmentation for training set only
    augmenter = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15, fill_mode='constant'),
        layers.RandomZoom((-0.1, 0.2)),
        layers.RandomContrast(0.1),
        layers.GaussianNoise(0.01)
    ])

    # Save processed images
    for ds, split in [(dataset, "train"), (val_ds, "val"), (test_ds, "test")]:
        for i, (images, labels) in enumerate(ds):
            for j, (image, label) in enumerate(zip(images, labels)):
                # Normalize
                image = normalization(image)

                # Apply augmentation only to training images
                if split == "train":
                    image = augmenter(image, training=True)

                class_name = "Normal" if label.numpy() == 0 else "Abnormal"
                file_name = f"{i}_{j}.jpg"
                save_image(image, os.path.join(save_dir, split, class_name), file_name)


###############################
# 3. Unlabeled Data Processing
###############################

def load_and_save_unlabeled_data(base_path, save_dir, img_size=(224, 224), batch_size=32):
    """
    Load, normalize, and save unlabeled medical images.

    Args:
        base_path: Path to unlabeled Train/ folder.
        save_dir: Path to save pseudo-labeled images.
    """
    unlabeled_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_path, "Train"),
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    normalization = layers.Rescaling(1./255)

    for i, images in enumerate(unlabeled_ds):
        for j, image in enumerate(images):
            image = normalization(image)
            file_name = f"{i}_{j}.jpg"
            save_image(image, os.path.join(save_dir, "unlabeled"), file_name)


###############################
# 4. Pseudo-Labeling & Storage
###############################

def generate_and_store_pseudo_labels(model, save_dir, confidence_threshold=0.9, temperature=0.7):
    """
    Generate and store pseudo-labels with temperature scaling.

    Args:
        model: Pre-trained model for prediction.
        save_dir: Folder to store pseudo-labeled images.
        confidence_threshold: Minimum confidence score.
        temperature: Softmax temperature for confidence calibration.
    """
    # Load preprocessed unlabeled data
    unlabeled_images = []
    file_names = []
    unlabeled_path = os.path.join(save_dir, "unlabeled")

    for img_name in os.listdir(unlabeled_path):
        img = cv2.imread(os.path.join(unlabeled_path, img_name))
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        unlabeled_images.append(img)
        file_names.append(img_name)

    unlabeled_images = np.array(unlabeled_images)

    # Predict labels
    probs = model.predict(unlabeled_images, verbose=0)
    scaled_probs = tf.nn.sigmoid(probs / temperature).numpy().flatten()
    
    for i, prob in enumerate(scaled_probs):
        if prob > confidence_threshold:
            pseudo_label = "Normal" if prob < 0.5 else "Abnormal"
            save_image(unlabeled_images[i], os.path.join(save_dir, "pseudo_labeled", pseudo_label), file_names[i])


###############################
# 5. Main Execution
###############################

def main():
    LABELED_PATH = "raw_data/labeled_data"
    UNLABELED_PATH = "raw_data/unlabeled_data"
    SAVE_DIR = "processed_data"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    print("🔄 Processing labeled images...")
    load_and_preprocess_data(LABELED_PATH, SAVE_DIR, IMG_SIZE, BATCH_SIZE)

    print("🔄 Processing unlabeled images...")
    load_and_save_unlabeled_data(UNLABELED_PATH, SAVE_DIR, IMG_SIZE, BATCH_SIZE)

    # Load model for pseudo-labeling
    model = tf.keras.models.load_model("your_pretrained_model.h5")
    
    print("🔄 Generating pseudo-labels...")
    generate_and_store_pseudo_labels(model, SAVE_DIR)

    print("\n✅ Preprocessing complete. Data saved in 'processed_data/'.")

if __name__ == "__main__":
    main()
