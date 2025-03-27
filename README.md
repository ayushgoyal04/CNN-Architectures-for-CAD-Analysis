# Comparative Analysis of CNN Architectures on Coronary Artery Disease Dataset

This repository contains the code and documentation for the research titled "Comparative Analysis of CNN Architectures on Coronary Artery Disease Dataset." The study compares various Convolutional Neural Network (CNN) architectures on a dataset of coronary artery images to identify the best-performing model for classification tasks. The study also explores the impact of pseudo-labeling techniques on enhancing the training dataset.

## Overview

This project aims to evaluate different CNN architectures, including ResNet, VGG, Inception, EfficientNet, DenseNet, MobileNet, and NASNet, using a dataset of coronary artery disease images. The study utilizes labeled and unlabeled data, where pseudo-labeling is applied to the unlabeled dataset to improve model performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Methodology](#methodology)
4. [Training and Evaluation](#training-and-evaluation)
5. [Expected Outcomes](#expected-outcomes)
6. [Future Work](#future-work)
7. [Increasing Dataset Size with Pseudo-Labeling](#increasing-dataset-size-with-pseudo-labeling)

## Introduction

Convolutional Neural Networks (CNNs) have become a crucial tool in medical imaging, particularly for classifying diseases from images. In this study, we aim to evaluate the effectiveness of multiple CNN models on a dataset containing coronary artery images labeled as normal or abnormal. Pseudo-labeling is employed to leverage a large amount of unlabeled data, enhancing the training dataset and potentially improving model performance.

## Dataset Overview

### Labeled Dataset
- **Normal Images:** 127
- **Abnormal Images:** 78
- **Total:** 205 images

### Unlabeled Dataset
- **Training:** 1000 images
- **Testing:** 300 images
- **Validation:** 200 images
- **Total:** 1500 images

## Methodology

### Data Preprocessing
- **Image Resizing:** All images are resized to 224x224 pixels.
- **Normalization:** Pixel values are scaled between 0 and 1.
- **Augmentation (Training Set Only):**
  - Rotation (15°)
  - Width/Height Shift (10%)
  - Horizontal Flip
  - Brightness Adjustment (80%–120%)
  - Zoom (20%)

### Pseudo-Labeling for Unlabeled Data
- A pre-trained CNN model is used to generate pseudo-labels for the unlabeled dataset.
- Only high-confidence predictions (>90% probability) are used to create pseudo-labels.
- The augmented training dataset includes both real and pseudo-labeled data.

### CNN Architectures for Comparison
- **ResNet**
- **VGG**
- **Inception**
- **EfficientNet**
- **DenseNet**
- **MobileNet**
- **NASNet**

## Training and Evaluation

Models will be trained on the expanded dataset (labeled + pseudo-labeled). The evaluation metrics for model performance include:
- Accuracy
- Precision & Recall
- F1-score
- ROC-AUC Curve

## Expected Outcomes

The expected outcomes of this research are:
- Identifying the best CNN model for classifying coronary artery disease images.
- Assessing the impact of pseudo-labeling on training performance.
- Providing insights into model generalization on medical imaging datasets.

## Future Work

- Extending the analysis to larger datasets.
- Experimenting with semi-supervised learning techniques.
- Deploying the best-performing model in a real-world diagnostic system.

## Increasing Dataset Size with Pseudo-Labeling

Pseudo-labeling will increase the size of the training dataset. Here's how:

### Original Training Set (Labeled)
- You started with **127 normal** + **78 abnormal** = 205 labeled images.
- After an **80-20 split**, you had **164 images** for training (80% of 205).

### Adding Pseudo-Labeled Data
- You have **1000 unlabeled training images**.
- Let's assume **60%** of them (600 images) have high-confidence pseudo-labels.

### New Training Set Size
- **Original Labeled Training Set:** 164 images
- **High-Confidence Pseudo-Labeled Data:** 600 images
- **New Training Set Total:** 164 + 600 = **764 images**

### Benefits of Increasing Dataset Size
- **Better Model Generalization:** Reduces overfitting on a small dataset.
- **More Training Data for CNNs:** Deep learning models perform better with more data.
- **Utilizes Unlabeled Data Effectively:** No manual labeling efforts are required.

### Will the Validation & Test Sets Increase?
No. Only the **training set** grows. The **validation** and **test sets** remain the same size for a fair comparison of different CNN models.

---

## How to Run the Code

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/CNN-Architectures-for-CAD-Analysis.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Preprocess the data:
    - Run the data preprocessing script to resize, normalize, and augment the dataset:
    ```bash
    python preprocess_data.py
    ```

4. Train the models:
    - Each model can be trained separately using the provided training scripts:
    ```bash
    python train_resnet.py
    python train_vgg.py
    python train_inception.py
    python train_efficientnet.py
    python train_densenet.py
    python train_mobilenet.py
    python train_nasnet.py
    ```

5. Evaluate the models:
    - After training, evaluate the performance of the models:
    ```bash
    python evaluate_model.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, fork, or open issues if you find bugs or have any suggestions for improvements!
