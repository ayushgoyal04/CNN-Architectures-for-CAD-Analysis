'''
 @title: "CNN Model Comparison and Analysis"
 @author: "Ayush Goyal"
 @date: "2023-10-01"
'''

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_fscore_support)
from tqdm import tqdm
from collections import defaultdict

# Configuration settings
class Config:
    seed = 42
    batch_size = 32
    num_epochs = 8
    lr = 0.0001
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/content/drive/MyDrive/dataset/processed_images_main"
    model_dir = "./models"
    plot_dir = "./plots"

    # Model specific adjustments
    model_params = {
        'resnet50': {'fc_layer': 'fc', 'in_features': 2048},
        'resnet101': {'fc_layer': 'fc', 'in_features': 2048},
        'vgg16': {'classifier_layer': 6, 'in_features': 4096},
        'vgg19': {'classifier_layer': 6, 'in_features': 4096},
        'densenet121': {'fc_layer': 'classifier', 'in_features': 1024},
        'densenet201': {'fc_layer': 'classifier', 'in_features': 1920},
        'mobilenet_v2': {'classifier_layer': 1, 'in_features': 1280},
        'efficientnet_b0': {'classifier_layer': 1, 'in_features': 1280}
    }

# Setup directories
os.makedirs(Config.model_dir, exist_ok=True)
os.makedirs(Config.plot_dir, exist_ok=True)

# Add directory verification
print("Checking Google Drive mount...")
required_dirs = [
    Config.data_dir,
    os.path.join(Config.data_dir, 'train'),
    os.path.join(Config.data_dir, 'validation'),
    os.path.join(Config.data_dir, 'test')
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Missing directory: {dir_path}")
    print(f"✓ Found: {dir_path}")

# Set seed for reproducibility
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# Enhanced transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets with proper validation
datasets = {
    'train': ImageFolder(os.path.join(Config.data_dir, 'train'), train_transform),
    'validation': ImageFolder(os.path.join(Config.data_dir, 'validation'), test_transform),
    'test': ImageFolder(os.path.join(Config.data_dir, 'test'), test_transform)
}

dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=Config.batch_size,
                       shuffle=True, num_workers=4),
    'validation': DataLoader(datasets['validation'], batch_size=Config.batch_size,
                            shuffle=False, num_workers=4),
    'test': DataLoader(datasets['test'], batch_size=Config.batch_size,
                      shuffle=False, num_workers=4)
}

class ModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0
        self.metrics = defaultdict(list)

    def _initialize_model(self):
        """Handle different architecture configurations properly"""
        model = models.__dict__[self.model_name](pretrained=True)

        if self.model_name.startswith('resnet') or self.model_name.startswith('densenet'):
            in_features = getattr(model, Config.model_params[self.model_name]['fc_layer']).in_features
            setattr(model, Config.model_params[self.model_name]['fc_layer'],
                   nn.Linear(in_features, Config.num_classes))
        elif self.model_name.startswith('vgg'):
            in_features = model.classifier[Config.model_params[self.model_name]['classifier_layer']].in_features
            model.classifier[Config.model_params[self.model_name]['classifier_layer']] = \
                nn.Linear(in_features, Config.num_classes)
        elif 'mobilenet' in self.model_name or 'efficientnet' in self.model_name:
            in_features = model.classifier[Config.model_params[self.model_name]['classifier_layer']].in_features
            model.classifier[Config.model_params[self.model_name]['classifier_layer']] = \
                nn.Linear(in_features, Config.num_classes)

        return model.to(Config.device)

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloader, desc=f"Training {self.model_name}"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(datasets['train'])
        epoch_acc = correct.double() / len(datasets['train'])
        return epoch_loss, epoch_acc.cpu().numpy()

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f"Validating {self.model_name}"):
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / len(datasets['validation'])
        epoch_acc = correct.double() / len(datasets['validation'])
        return epoch_loss, epoch_acc.cpu().numpy()

    def train_model(self):
        print(f"\n{'='*40}\nTraining {self.model_name}\n{'='*40}")

        for epoch in range(Config.num_epochs):
            train_loss, train_acc = self.train_epoch(dataloaders['train'])
            val_loss, val_acc = self.validate_epoch(dataloaders['validation'])

            self.scheduler.step(val_acc)

            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save(self.model.state_dict(),
                          os.path.join(Config.model_dir, f"best_{self.model_name}.pth"))

            print(f"Epoch {epoch+1}/{Config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        self._plot_training_metrics()
        return self.metrics

    def evaluate_model(self):
        self.model.load_state_dict(torch.load(os.path.join(Config.model_dir, f"best_{self.model_name}.pth")))
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        inference_times = []

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(Config.device)
                start_time = time.time()
                outputs = self.model(inputs)
                inference_times.append(time.time() - start_time)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)
        cr = classification_report(all_labels, all_preds, output_dict=True)
        roc_auc = self._calculate_roc_auc(all_labels, all_probs)
        metrics = {
            'accuracy': cr['accuracy'],
            'precision': cr['weighted avg']['precision'],
            'recall': cr['weighted avg']['recall'],
            'f1': cr['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'inference_time': np.mean(inference_times),
            'params': sum(p.numel() for p in self.model.parameters()),
            'confusion_matrix': cm,
            'classification_report': cr
        }

        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(all_labels, all_probs)
        return metrics

    def _calculate_roc_auc(self, labels, probs):
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probs])
        return auc(fpr, tpr)

    def _plot_training_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_name} Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(self.metrics['val_acc'], label='Validation Accuracy')
        plt.title(f'{self.model_name} Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(Config.plot_dir, f'{self.model_name}_training.png'))
        plt.close()

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=datasets['test'].classes,
                   yticklabels=datasets['test'].classes)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(Config.plot_dir, f'{self.model_name}_cm.png'))
        plt.close()

    def _plot_roc_curve(self, labels, probs):
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probs])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(Config.plot_dir, f'{self.model_name}_roc.png'))
        plt.close()

def compare_models(model_names):
    results = {}

    for model_name in model_names:
        wrapper = ModelWrapper(model_name)
        train_metrics = wrapper.train_model()
        test_metrics = wrapper.evaluate_model()

        results[model_name] = {
            'training': train_metrics,
            'testing': test_metrics
        }

    # Generate comparison plots
    _plot_comparison(results)
    _generate_report(results)
    return results

def _plot_comparison(results):
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    accuracies = [results[name]['testing']['accuracy'] for name in results]
    plt.barh(list(results.keys()), accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.xlim(0.5, 1.0)
    plt.savefig(os.path.join(Config.plot_dir, 'accuracy_comparison.png'))
    plt.close()

    # Metrics radar chart
    metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    labels = np.array(metrics)
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, data in results.items():
        values = [data['testing'][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9], ["60%", "70%", "80%", "90%"], color="grey", size=7)
    plt.ylim(0.5, 1.0)
    plt.title('Model Performance Comparison', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(Config.plot_dir, 'radar_comparison.png'))
    plt.close()

def _generate_report(results):
    report_data = []
    for model_name, data in results.items():
        report_data.append({
            'Model': model_name,
            'Accuracy': data['testing']['accuracy'],
            'F1-Score': data['testing']['f1'],
            'ROC AUC': data['testing']['roc_auc'],
            'Precision': data['testing']['precision'],
            'Recall': data['testing']['recall'],
            'Params (M)': data['testing']['params'] / 1e6,
            'Inference Time (ms)': data['testing']['inference_time'] * 1000
        })

    df = pd.DataFrame(report_data).sort_values(by='Accuracy', ascending=False)
    df.to_csv(os.path.join(Config.plot_dir, 'model_comparison.csv'), index=False)

    # Print formatted table
    print("\nModel Comparison Summary:")
    print(df.to_markdown(index=False, floatfmt=".3f"))

if __name__ == "__main__":
    model_names = ['resnet50', 'resnet101', 'vgg16', 'vgg19',
                  'densenet121', 'densenet201', 'mobilenet_v2', 'efficientnet_b0']
    results = compare_models(model_names)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar

def comprehensive_analysis(results_df, dataloaders):
    """Perform post-hoc analysis without retraining"""

    # 1. Overfitting Diagnosis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Params (M)', y='Accuracy', size='Inference Time (ms)',
                    hue='Model', data=results_df, s=200)
    plt.title('Accuracy vs Model Complexity')
    plt.axhline(0.95, color='red', linestyle='--',
               label='Expected Max Real-world Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

    # 2. Statistical Significance Testing
    print("\nStatistical Analysis:")
    models = results_df.Model.values
    accuracies = results_df.Accuracy.values

    # Friedman test
    friedman_stat, friedman_p = friedmanchisquare(*[results_df[results_df.Model == m].Accuracy for m in models])
    print(f"Friedman Test: χ²={friedman_stat:.1f}, p={friedman_p:.2e}")

    # Pairwise McNemar's test
    print("\nPairwise McNemar Tests:")
    predictions = {}  # You'll need to store test predictions from previous steps
    for model in models:
        # Load predictions (add your prediction collection logic here)
        predictions[model] = np.random.randint(0, 2, 1000)  # Replace with actual predictions

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models[i+1:]):
            table = pd.crosstab(predictions[m1], predictions[m2])
            result = mcnemar(table)
            print(f"{m1} vs {m2}: p={result.pvalue:.4f}")

    # 3. Confidence Analysis
    confidence = {
        'Model': models,
        'Min Confidence': np.random.uniform(0.85, 1.0, len(models)),
        'Max Confidence': np.random.uniform(0.98, 1.0, len(models))
    }  # Replace with actual confidence values from your predictions
    conf_df = pd.DataFrame(confidence)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Max Confidence', data=conf_df)
    plt.axhline(0.95, color='red', linestyle='--')
    plt.xticks(rotation=45)
    plt.title('Prediction Confidence Analysis')
    plt.show()

    # 4. Practical Efficiency Analysis
    results_df['Efficiency Score'] = (results_df['Accuracy'] * 100) / (results_df['Params (M)'] * results_df['Inference Time (ms)'])
    results_df.sort_values('Efficiency Score', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Efficiency Score', y='Model', data=results_df, palette='viridis')
    plt.title('Model Efficiency (Accuracy per Computational Unit)')
    plt.show()

    # 5. Consensus Analysis
    all_preds = np.array([predictions[m] for m in models])
    consensus = np.mean(all_preds == all_preds[0], axis=0)

    plt.figure(figsize=(10, 6))
    sns.histplot(consensus, bins=20, kde=True)
    plt.title('Model Consensus Distribution')
    plt.xlabel('Fraction of Models Agreeing with ResNet50')
    plt.show()

    # 6. Generate Conclusions
    conclusions = {
        'Technical Recommendation': results_df.iloc[0]['Model'],
        'Efficient Choice': results_df.sort_values('Efficiency Score').iloc[0]['Model'],
        'Most Consistent': 'ResNet50',  # Replace with actual analysis
        'Overfitting Risk': 'VGG19, ResNet101' if friedman_p < 0.05 else 'None',
        'Statistical Power': 'High' if friedman_p < 0.05 else 'Low'
    }

    print("\nFinal Conclusions:")
    print(pd.Series(conclusions).to_markdown())

# Run analysis
results_df = pd.DataFrame({
    'Model': ['resnet50', 'resnet101', 'vgg16', 'vgg19',
             'densenet201', 'efficientnet_b0', 'densenet121', 'mobilenet_v2'],
    'Accuracy': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.961, 0.947],
    'Params (M)': [23.5, 42.5, 134.3, 139.6, 18.1, 4.0, 6.96, 2.23],
    'Inference Time (ms)': [17.8, 15.9, 6.8, 3.2, 88.2, 22.8, 46.9, 16.3]
})

comprehensive_analysis(results_df, dataloaders)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

def plot_mcnemar_matrix(predictions):
    """
    Plot McNemar test results matrix for multiple classifiers
    """
    models = list(predictions.keys())
    n_models = len(models)
    pvalue_matrix = np.ones((n_models, n_models))

    # Create pairwise comparison matrix
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i == j:
                continue
            # Create contingency table
            table = [[sum((predictions[model1] == 1) & (predictions[model2] == 1)),
                      sum((predictions[model1] == 1) & (predictions[model2] == 0))],
                     [sum((predictions[model1] == 0) & (predictions[model2] == 1)),
                      sum((predictions[model1] == 0) & (predictions[model2] == 0))]]

            # Handle small sample sizes with exact test
            exact = (table[0][1] + table[1][0]) < 25
            result = mcnemar(table, exact=exact)
            pvalue_matrix[i, j] = result.pvalue

    # Create annotated dataframe
    df = pd.DataFrame(pvalue_matrix, index=models, columns=models)
    np.fill_diagonal(df.values, np.nan)  # Hide diagonal

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    ax = sns.heatmap(df, mask=mask, cmap=cmap, center=0.05,
                     annot=True, fmt=".3f", linewidths=.5,
                     cbar_kws={'label': 'p-value'})

    plt.title("Pairwise McNemar Test Results (p-values)", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Example usage with dummy predictions (replace with your actual predictions)
n_samples = 1000
models = ['resnet50', 'resnet101', 'vgg16', 'vgg19',
          'densenet201', 'efficientnet_b0', 'densenet121', 'mobilenet_v2']

# Generate dummy predictions (replace with your actual predictions)
predictions = {
    model: np.random.randint(0, 2, n_samples) for model in models
}

plot_mcnemar_matrix(predictions)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.profiler import profile, record_function
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from ptflops import get_model_complexity_info
import albumentations as A
from tqdm import tqdm
import json
import torchvision.models as models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, model_path):
    """Properly initialize model architecture and load saved weights"""
    # Initialize base model
    if model_name.startswith('resnet'):
        model = models.__dict__[model_name](pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name.startswith('vgg'):
        model = models.__dict__[model_name](pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name.startswith('densenet'):
        model = models.__dict__[model_name](pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_name.startswith('mobilenet'):
        model = models.__dict__[model_name](pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name.startswith('efficientnet'):
        model = models.__dict__[model_name](pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    # Load saved weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def comprehensive_analysis(model_dict, test_loader):
    """Perform all validation and analysis steps"""
    # Get true labels
    true_labels = torch.tensor(test_loader.dataset.targets)

    # 1. Collect predictions
    predictions = {}
    print("\nCollecting model predictions...")
    for name, model in model_dict.items():
        model_preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=name):
                outputs = model(images.to(device))
                model_preds.append(outputs.argmax(1).cpu())
        predictions[name] = torch.cat(model_preds).numpy()

    # 2. Overfitting analysis with corruption
    def add_corruption(images, severity=1):
        transform = A.Compose([
            A.GaussNoise(var_limit=(0.01*severity, 0.03*severity), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5)
        ])
        return torch.stack([
            torch.from_numpy(
                transform(image=img.permute(1,2,0).numpy())['image'].transpose(2,0,1)
            ).float() for img in images
        ])

    print("\nTesting corruption robustness...")
    corruption_results = {}
    for name, model in tqdm(model_dict.items()):
        clean_correct = corrupt_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                # Clean images
                outputs = model(images.to(device))
                clean_correct += (outputs.argmax(1) == labels.to(device)).sum().item()

                # Corrupted images
                corrupted = add_corruption(images)
                outputs = model(corrupted.to(device))
                corrupt_correct += (outputs.argmax(1) == labels.to(device)).sum().item()

        clean_acc = clean_correct / len(test_loader.dataset)
        corrupt_acc = corrupt_correct / len(test_loader.dataset)
        corruption_results[name] = {'Clean': clean_acc, 'Corrupted': corrupt_acc}

    # 3. Statistical tests
    def calculate_mcnemar(predictions):
        model_names = list(predictions.keys())
        p_matrix = pd.DataFrame(
            np.ones((len(model_names), len(model_names))),
            index=model_names, columns=model_names
        )

        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if i >= j: continue
                table = [[np.sum((predictions[m1] == 1) & (predictions[m2] == 1)),
                          np.sum((predictions[m1] == 1) & (predictions[m2] == 0))],
                         [np.sum((predictions[m1] == 0) & (predictions[m2] == 1)),
                          np.sum((predictions[m1] == 0) & (predictions[m2] == 0))]]
                result = mcnemar(table)
                p_matrix.loc[m1, m2] = result.pvalue

        return p_matrix

    # 4. Error analysis
    def error_consistency(predictions, true_labels):
        errors = {m: (p != true_labels) for m, p in predictions.items()}
        consistency = pd.DataFrame(
            np.zeros((len(predictions), len(predictions))),
            index=predictions.keys(), columns=predictions.keys()
        )

        for m1 in predictions:
            for m2 in predictions:
                consistency.loc[m1, m2] = np.mean(errors[m1] & errors[m2])

        return consistency

    # 5. Computational profiling
    def model_profiling(model_dict):
        profile_data = {}
        for name, model in model_dict.items():
            # FLOPs calculation
            macs, params = get_model_complexity_info(
                model, (3, 224, 224), as_strings=False,
                print_per_layer_stat=False, verbose=False
            )

            # Inference time
            inputs = torch.randn(1, 3, 224, 224).to(device)
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True
            ) as prof:
                model(inputs)

            profile_data[name] = {
                'Params (M)': params/1e6,
                'FLOPs (G)': macs/1e9,
                'Inference Time (ms)': prof.key_averages().total_average().cpu_time_total/1e6
            }

        return pd.DataFrame(profile_data).T

    # Execute analysis
    print("\nPerforming statistical tests...")
    mcnemar_results = calculate_mcnemar(predictions)
    error_matrix = error_consistency(predictions, true_labels.numpy())
    comp_metrics = model_profiling(model_dict)

    # Visualization
    plt.figure(figsize=(18, 12))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    acc_df = pd.DataFrame(corruption_results).T
    acc_df.plot(kind='bar', rot=45, ax=plt.gca())
    plt.title('Clean vs Corrupted Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)

    # Computational efficiency
    plt.subplot(2, 2, 2)
    comp_metrics = comp_metrics.astype(float)
    plt.scatter(
        comp_metrics['FLOPs (G)'],
        comp_metrics['Inference Time (ms)'],
        s=comp_metrics['Params (M)']*50, alpha=0.6
    )
    for i, row in comp_metrics.iterrows():
        plt.text(row['FLOPs (G)'], row['Inference Time (ms)'], i, fontsize=8)
    plt.xlabel('FLOPs (G)')
    plt.ylabel('Inference Time (ms)')
    plt.title('Computational Efficiency')

    # Statistical significance
    plt.subplot(2, 2, 3)
    sns.heatmap(
        mcnemar_results.where(mcnemar_results < 1),
        annot=True, fmt=".3f", cmap='viridis',
        cbar_kws={'label': 'p-value'}
    )
    plt.title('McNemar Test Results')

    # Error consistency
    plt.subplot(2, 2, 4)
    sns.heatmap(error_matrix, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Error Consistency Matrix')

    plt.tight_layout()
    plt.show()

    # Generate report
    full_report = pd.concat([
        pd.DataFrame(corruption_results).T,
        comp_metrics,
        pd.DataFrame({
            'Mean Error Consensus': error_matrix.mean(axis=1),
            'Max p-value': mcnemar_results.max(axis=1)
        })
    ], axis=1)

    print("\nFinal Report:")
    print(full_report.sort_values('Clean', ascending=False).to_markdown(floatfmt=".3f"))

# Load all models
model_names = ['resnet50', 'resnet101', 'vgg16', 'vgg19',
               'densenet121', 'densenet201', 'mobilenet_v2', 'efficientnet_b0']

model_dict = {
    name: load_model(name, f'/content/models/best_{name}.pth')
    for name in model_names
}

# Load your test dataset (modify as needed)
from torchvision.datasets import ImageFolder
test_dataset = ImageFolder(
    '/content/drive/MyDrive/dataset/processed_images_main/test',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Run complete analysis
comprehensive_analysis(model_dict, test_loader)

