"""
Medical Image Pipeline with Pseudo-Labeling
Author: Ayush  
Date: 2025-03-27

Features:
- Proper medical imaging data handling
- Leakage-proof dataset splitting
- Temperature-scaled pseudo-labeling
- Class-balanced training
- GPU-optimized preprocessing
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

###############################
# 1. Data Loading & Preparation
###############################

def load_labeled_data(base_path: str, img_size: tuple = (224, 224), batch_size: int = 32):
    """
    Load and split labeled medical images with proper class balancing.

    Args:
        base_path: Path to directory containing Normal/Abnormal subfolders.
        img_size: Target image dimensions (height, width).
        batch_size: Number of images per batch.

    Returns:
        tuple: (train_ds, val_ds, test_ds), class_weights
    """
    # Train set (70%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_path,
        label_mode='int',  # Ensure integer labels for class weighting
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42,
        validation_split=0.3,
        subset='training'
    )

    # Temp set (30%) - Split into validation (15%) and test (15%)
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

    # Compute class weights
    class_counts = np.bincount(np.concatenate([y.numpy() for _, y in train_ds], axis=0))
    class_weights = {0: class_counts[1] / class_counts.sum(), 1: class_counts[0] / class_counts.sum()}

    # Normalization
    normalization = layers.Rescaling(1./255)

    return (
        train_ds.map(lambda x, y: (normalization(x), y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE),
        val_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(tf.data.AUTOTUNE),
        test_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(tf.data.AUTOTUNE),
        class_weights
    )

###############################
# 2. Medical Image Augmentation
###############################

def medical_augmentation():
    """
    Create augmentation pipeline for medical images.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15, fill_mode='constant'),
        layers.RandomZoom((-0.1, 0.2)),
        layers.RandomContrast(0.1),
        layers.GaussianNoise(0.01)
    ], name='medical_augmentation')

###############################
# 3. Unlabeled Data Processing
###############################

def load_unlabeled_data(base_path: str, img_size: tuple = (224, 224), batch_size: int = 32):
    """
    Load and normalize unlabeled medical images.

    Args:
        base_path: Root directory containing Train subfolder.

    Returns:
        tf.data.Dataset
    """
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_path, "Train"),
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    ).map(layers.Rescaling(1./255))

###############################
# 4. Pseudo-Labeling
###############################

def generate_pseudo_labels(model, unlabeled_ds, confidence_threshold=0.9, temperature=0.7):
    """
    Generate pseudo-labels with temperature scaling.

    Args:
        model: Pre-trained model for prediction.
        unlabeled_ds: Dataset of unlabeled images.
        confidence_threshold: Minimum confidence score.
        temperature: Softmax temperature for confidence calibration.

    Returns:
        Dataset with (image, pseudo_label) pairs.
    """
    # Get predictions
    probs = model.predict(unlabeled_ds, verbose=0)
    scaled_probs = tf.nn.sigmoid(probs / temperature).numpy().flatten()
    
    # Filter high-confidence samples
    confidence_mask = scaled_probs > confidence_threshold
    images = np.concatenate([x.numpy() for x in unlabeled_ds], axis=0)
    filtered_images = images[confidence_mask]
    pseudo_labels = (scaled_probs[confidence_mask] > 0.5).astype(np.float32)

    return tf.data.Dataset.from_tensor_slices((filtered_images, pseudo_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

###############################
# 5. Complete Pipeline
###############################

def main_pipeline():
    # Configuration
    LABELED_PATH = "labeled_data"
    UNLABELED_PATH = "unlabeled_data"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.9

    # Load labeled data
    train_ds, val_ds, test_ds, class_weights = load_labeled_data(LABELED_PATH, IMG_SIZE, BATCH_SIZE)

    # Apply augmentation to training data
    augmenter = medical_augmentation()
    augmented_train = train_ds.map(lambda x, y: (augmenter(x, training=True), y)).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    # Load and initialize the model
    base_model = tf.keras.applications.EfficientNetV2S(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.load_weights("your_pretrained_weights.h5")

    # Load and generate pseudo-labels
    unlabeled_ds = load_unlabeled_data(UNLABELED_PATH, IMG_SIZE, BATCH_SIZE)
    pseudo_ds = generate_pseudo_labels(model, unlabeled_ds, CONFIDENCE_THRESHOLD)

    # Combine datasets
    final_train = augmented_train.concatenate(pseudo_ds).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    print("\nPipeline Summary:")
    print(f"Training samples: {len(train_ds)*BATCH_SIZE}")
    print(f"Pseudo-labels: {len(pseudo_ds)*BATCH_SIZE}")
    print(f"Final training size: {len(final_train)*BATCH_SIZE}")
    print(f"Class weights: {class_weights}")

    return final_train, val_ds, test_ds, class_weights

if __name__ == "__main__":
    final_train, val_ds, test_ds, class_weights = main_pipeline()
