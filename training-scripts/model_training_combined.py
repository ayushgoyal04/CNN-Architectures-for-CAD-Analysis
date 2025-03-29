import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "dataset"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Data loaders (Windows users: `num_workers=0` to avoid multiprocessing issues)
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
    "val": DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0),
    "test": DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
}

# Define models to compare
model_names = [
    "resnet50", "resnet101", "vgg16", "vgg19", "densenet121", "densenet201",
    "mobilenet_v2", "mobilenet_v3_large", "nasnet_mobile", "efficientnet_b0"
]

def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name in ["vgg16", "vgg19"]:
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name in ["densenet121", "densenet201"]:
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_name in ["mobilenet_v2", "mobilenet_v3_large"]:
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == "nasnet_mobile":
        model = models.mnasnet1_0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)

def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloaders['train'])}")

def evaluate_model(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    return acc, prec, rec, f1, auc

if __name__ == "__main__":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    results = []

    for model_name in model_names:
        print(f"Training {model_name}...")
        model = get_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_model(model, criterion, optimizer, num_epochs=10)

        acc, prec, rec, f1, auc = evaluate_model(model)
        results.append([model_name, acc, prec, rec, f1, auc])

    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    df.to_csv(os.path.join(log_dir, 'model_comparison.csv'), index=False)
    print("Results saved to logs/model_comparison.csv")
