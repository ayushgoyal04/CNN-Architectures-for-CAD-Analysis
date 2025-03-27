# CNN Models Input and Output Format Analysis

This document provides an analysis of the input and output formats of various models from each CNN family. The goal is to ensure consistency in the research process.

## 📌 **ResNet Family**

### 1️⃣ ResNet50

- **Input:** `(224, 224, 3)` RGB images
- **Output:** 1000-class softmax (ImageNet) or modified for binary classification

### 2️⃣ ResNet101

- **Input:** `(224, 224, 3)`
- **Output:** Same as ResNet50

#### Verdict: 
✅ **No issue** – standard CNN input and output.

---

## 📌 **VGG Family**

### 1️⃣ VGG16

- **Input:** `(224, 224, 3)`
- **Output:** 1000-class softmax (modifiable for binary classification)

### 2️⃣ VGG19

- **Input:** `(224, 224, 3)`
- **Output:** Same as VGG16

#### Verdict:
✅ **No issue** – standard CNN format.

---

## 📌 **Inception Family**

### 1️⃣ InceptionV3

- **Input:** `(299, 299, 3)` (🚨 Different size)
- **Output:** 1000-class softmax

### 2️⃣ InceptionResNetV2

- **Input:** `(299, 299, 3)` (🚨 Different size)
- **Output:** Same as InceptionV3

#### Issue:
⚠️ **Different input size** (299x299 instead of 224x224)

#### Possible Fix:
Resize all images to `(299, 299, 3)` before training.

---

## 📌 **EfficientNet Family**

### 1️⃣ EfficientNetB0

- **Input:** `(224, 224, 3)`
- **Output:** 1000-class softmax

### 2️⃣ EfficientNetB3

- **Input:** `(300, 300, 3)` (🚨 Different size)
- **Output:** Same as EfficientNetB0

#### Issue:
⚠️ **Different input size** (300x300 instead of 224x224)

#### Possible Fix:
Resize images to EfficientNet’s required size.

---

## 📌 **DenseNet Family**

### 1️⃣ DenseNet121

- **Input:** `(224, 224, 3)`
- **Output:** 1000-class softmax

### 2️⃣ DenseNet201

- **Input:** `(224, 224, 3)`
- **Output:** Same as DenseNet121

#### Verdict:
✅ **No issue** – standard CNN format.

---

## 📌 **MobileNet Family**

### 1️⃣ MobileNetV2

- **Input:** `(224, 224, 3)`
- **Output:** 1000-class softmax

### 2️⃣ MobileNetV3-Large

- **Input:** `(224, 224, 3)`
- **Output:** Same as MobileNetV2

#### Verdict:
✅ **No issue** – standard CNN format.

---

## 📌 **NASNet Family**

### 1️⃣ NASNetMobile

- **Input:** `(224, 224, 3)`
- **Output:** 1000-class softmax

### 2️⃣ NASNetLarge

- **Input:** `(331, 331, 3)` (🚨 Very different size)
- **Output:** 1000-class softmax

#### Issue:
⚠️ **NASNetLarge uses 331x331 input** (may require extra computation).

#### Possible Fix:
Only use **NASNetMobile** for consistency.

---

## 🚀 **Conclusion: Which Models Fit Well?**

### ✅ **Safe to use:**
- ResNet
- VGG
- DenseNet
- MobileNet
- NASNetMobile

### ⚠️ **Require image resizing:**
- Inception
- EfficientNet

### ❌ **Might be problematic:**
- NASNetLarge (uses 331x331)
