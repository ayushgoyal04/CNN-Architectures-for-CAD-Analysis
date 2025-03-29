import os
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "preprocessed_images_main"
results_dir = "results"
logs_dir = "logs"
saved_models_dir = "saved_models"

# Ensure directories exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(saved_models_dir, exist_ok=True)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
datasets_dict = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform)
    for x in ["train", "validation", "test"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=32, shuffle=True, num_workers=4)
    for x in ["train", "validation", "test"]
}

def train_and_evaluate_model(model, model_name, criterion, optimizer, num_epochs=10):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / len(datasets_dict[phase])
            epoch_acc = correct.double() / total

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.cpu().numpy())

            print(f"Epoch {epoch+1}/{num_epochs} - {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"{saved_models_dir}/{model_name}.pth")

    return history

def evaluate_model(model, model_name):
    model.load_state_dict(torch.load(f"{saved_models_dir}/{model_name}.pth"))
    model.eval()
    y_true, y_pred = [], []

    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{logs_dir}/{model_name}_classification_report.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{results_dir}/{model_name}_confusion_matrix.png")
    plt.close()

# Define models and hyperparameters
models_dict = {
    "ResNet50": models.resnet50(pretrained=True),
    "ResNet101": models.resnet101(pretrained=True),
    "VGG16": models.vgg16(pretrained=True),
    "VGG19": models.vgg19(pretrained=True),
    "DenseNet121": models.densenet121(pretrained=True),
    "DenseNet201": models.densenet201(pretrained=True),
    "MobileNetV2": models.mobilenet_v2(pretrained=True),
    "MobileNetV3-Large": models.mobilenet_v3_large(pretrained=True),
    "NASNetMobile": models.nasnetamobile(pretrained=True),
    "EfficientNetB0": models.efficientnet_b0(pretrained=True)
}

# Train and evaluate each model
results = []

for model_name, model in models_dict.items():
    print(f"Training {model_name}...")
    model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming binary classification
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_and_evaluate_model(model, model_name, criterion, optimizer)
    evaluate_model(model, model_name)

    results.append({
        "Model": model_name,
        "Best Validation Accuracy": max(history["val_acc"])
    })

df_results = pd.DataFrame(results)
df_results.to_csv(f"{logs_dir}/model_comparisons.csv", index=False)
print("Training complete. Results saved in logs/model_comparisons.csv")
