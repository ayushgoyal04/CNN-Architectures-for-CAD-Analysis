"""
Train and Compare Multiple CNN Models Using Transfer Learning

This script performs the following:
1. Loads and preprocesses images from the dataset (organized in train/validation/test folders).
2. Uses transfer learning to fine-tune various CNN models:
   - ResNet50, ResNet101
   - VGG16, VGG19
   - DenseNet121, DenseNet201
   - MobileNetV2, MobileNetV3-Large
   - NASNetMobile
   - EfficientNetB0
3. Trains each model while logging training and validation loss/accuracy.
4. Evaluates models on the test set using metrics: Accuracy, Precision, Recall, F1-score, AUC.
5. Generates detailed outputs including:
   - Loss and accuracy curves per epoch (saved as PNG images)
   - Confusion matrix heatmaps for test set predictions
   - Classification reports (saved as CSV files)
   - A comparison CSV file summarizing all models’ performance.
6. All outputs (plots, logs, saved model weights) are stored in designated folders for reproducibility.

Ensure you have installed the required libraries:
    pip install torch torchvision pandas scikit-learn matplotlib seaborn

Author: [Your Name]
Date: [Today's Date]
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# Set Device and Directories
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths (adjust these paths as necessary)
data_dir = "../processed_images_main"
logs_dir = "../logs"
results_dir = "../results"
saved_models_dir = "../saved_models"

# Create output directories if they do not exist
for directory in [logs_dir, results_dir, saved_models_dir]:
    os.makedirs(directory, exist_ok=True)

# -------------------------
# Define Data Transformations & Load Data
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets (expects train, validation, test folders with class subfolders)
datasets_dict = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform)
                 for x in ['train', 'validation', 'test']}

# Create DataLoaders
dataloaders = {x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x == 'train'), num_workers=4)
               for x in ['train', 'validation', 'test']}

# -------------------------
# Define Training Function
# -------------------------
def train_model(model, model_name, criterion, optimizer, num_epochs=10):
    """
    Train the given model and log training & validation metrics.
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best model based on validation accuracy
            if phase == 'validation' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(saved_models_dir, f"{model_name}.pth"))

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return history

# -------------------------
# Define Evaluation Function
# -------------------------
def evaluate_model(model, model_name):
    """
    Evaluate the trained model on the test set and generate analysis outputs.
    """
    # Load best saved weights
    model.load_state_dict(torch.load(os.path.join(saved_models_dir, f"{model_name}.pth")))
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = None

    # Save classification report
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(cls_report).transpose()
    report_df.to_csv(os.path.join(logs_dir, f"{model_name}_classification_report.csv"))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    metrics = {"Accuracy": accuracy, "Precision": precision,
               "Recall": recall, "F1 Score": f1, "AUC": auc}
    print(f"Evaluation Metrics for {model_name}: {metrics}")
    return metrics

# -------------------------
# Function to Plot Training Curves
# -------------------------
def plot_training_history(history, model_name):
    """
    Plot and save training and validation loss and accuracy curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{model_name}_loss_curve.png"))
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{model_name}_accuracy_curve.png"))
    plt.close()

# -------------------------
# Main Function: Train, Evaluate & Analyze All Models
# -------------------------
if __name__ == "__main__":
    model_names = [
        "ResNet50", "ResNet101",
        "VGG16", "VGG19",
        "DenseNet121", "DenseNet201",
        "MobileNetV2", "MobileNetV3-Large",
        "NASNetMobile", "EfficientNetB0"
    ]

    results_summary = []

    for model_name in model_names:
        print(f"\n\n===== Training {model_name} =====")
        # Initialize model based on its architecture
        if model_name in ["ResNet50", "ResNet101"]:
            model = models.__dict__[model_name.lower()](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif model_name in ["VGG16", "VGG19"]:
            model = models.__dict__[model_name.lower()](pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        elif model_name in ["DenseNet121", "DenseNet201"]:
            model = models.__dict__[model_name.lower()](pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, 2)
        elif model_name == "MobileNetV2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        elif model_name == "MobileNetV3-Large":
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        elif model_name == "NASNetMobile":
            # In torchvision, use mnasnet1_0 as a proxy for NASNetMobile
            model = models.mnasnet1_0(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, 2)
        elif model_name == "EfficientNetB0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            print(f"Model {model_name} not recognized.")
            continue

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train the model and log history
        history = train_model(model, model_name, criterion, optimizer, num_epochs=10)
        plot_training_history(history, model_name)
        metrics = evaluate_model(model, model_name)
        metrics['Model'] = model_name
        results_summary.append(metrics)

    # Save final comparison of all models to CSV
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(logs_dir, "model_comparisons.csv"), index=False)
    print("\nTraining & Evaluation complete. Detailed results are saved in the logs and results directories.")
