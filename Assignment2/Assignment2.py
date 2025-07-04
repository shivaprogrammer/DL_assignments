# -*- coding: utf-8 -*-
"""M24CSA029_M24CSA033.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PPocx_-sBkQ9U0x4MZmveJ_XPs-v3pR9

**Deep Learning - Assignment 2**

**Submitted By**

*   Shivani Tiwari (M24CSA029)
*   Suvigya Sharma (M24CSA033)


**Objective:** The goal of this assignment is to build a single Convolutional Neural Network (CNN) in PyTorch that can classify images from the CIFAR-100 dataset into three different categories at the same time:
*   Fine-level classification – Identify the exact object from 100 possible classes.
*   Superclass classification – Group the object into one of 20 broader categories.
*   Synthesized group classification – Further categorize the object into one of 9 custom-defined groups based on shared characteristics (e.g., vehicles, aquatic animals, plants, etc.).

Importing Libraries
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
import random

"""Setting Random Seed for Reproducibility"""

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

"""Custom CIFAR-100 Dataset with Superclass and Group Mapping"""

# CIFAR-100 dataset with custom group mapping
class CIFAR100_Custom(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar100 = datasets.CIFAR100(root=root, train=train, transform=transform, download=download)
        self.classes = self.cifar100.classes

        # Superclass mapping
        self.superclass_mapping = {
            "aquatic mammals": 3, "fish": 3, "flowers": 0, "food containers": 5,
            "fruit and vegetables": 0, "household electrical devices": 5, "household furniture": 5,
            "insects": 2, "large carnivores": 4, "large man-made outdoor things": 5,
            "large natural outdoor scenes": 8, "large omnivores and herbivores": 4,
            "medium-sized mammals": 7, "non-insect invertebrates": 2, "people": 6,
            "reptiles": 7, "small mammals": 7, "trees": 0, "vehicles 1": 1, "vehicles 2": 1
        }

        # Get superclass labels by dividing fine labels by 5
        self.superclass_labels = [self.cifar100.targets[i] // 5 for i in range(len(self.cifar100.targets))]
        self.group_labels = [self.superclass_mapping[self.get_superclass_name(label)] for label in self.superclass_labels]
        print("Unique group labels:", set(self.group_labels))

    def get_superclass_name(self, superclass_index):
        superclass_names = [
            "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
            "household electrical devices", "household furniture", "insects", "large carnivores",
            "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
            "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
            "trees", "vehicles 1", "vehicles 2"
        ]
        return superclass_names[superclass_index]

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        img, class_label = self.cifar100[idx]
        superclass_label = self.superclass_labels[idx]
        group_label = self.group_labels[idx]
        return img, class_label, superclass_label, group_label

"""Model Architecture"""

# CNN Architecture Model
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.class_head = nn.Linear(512, 100)
        self.superclass_head = nn.Linear(512, 20)
        self.group_head = nn.Linear(512, 9)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        class_output = self.class_head(x)
        superclass_output = self.superclass_head(x)
        group_output = self.group_head(x)
        return class_output, superclass_output, group_output

"""Training and Evaluation Functions with Confusion Matrix Visualization"""

# Training and evaluation functions
def train_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cuda'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, class_labels, superclass_labels, group_labels in dataloader:
            images, class_labels, superclass_labels, group_labels = images.to(device), class_labels.to(device), superclass_labels.to(device), group_labels.to(device)
            optimizer.zero_grad()
            class_output, superclass_output, group_output = model(images)
            loss = criterion(class_output, class_labels) + criterion(superclass_output, superclass_labels) + criterion(group_output, group_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

def evaluate_model(model, dataloader, criterion, device='cuda'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total, correct_class, correct_superclass, correct_group = 0, 0, 0, 0
    class_losses, superclass_losses, group_losses = [], [], []
    class_preds_all, superclass_preds_all, group_preds_all = [], [], []
    class_labels_all, superclass_labels_all, group_labels_all = [], [], []

    with torch.no_grad():
        for images, class_labels, superclass_labels, group_labels in dataloader:
            images, class_labels, superclass_labels, group_labels = (
                images.to(device), class_labels.to(device), superclass_labels.to(device), group_labels.to(device)
            )

            class_output, superclass_output, group_output = model(images)
            _, class_preds = torch.max(class_output, 1)
            _, superclass_preds = torch.max(superclass_output, 1)
            _, group_preds = torch.max(group_output, 1)

            class_losses.append(criterion(class_output, class_labels).item())
            superclass_losses.append(criterion(superclass_output, superclass_labels).item())
            group_losses.append(criterion(group_output, group_labels).item())

            total += class_labels.size(0)
            correct_class += (class_preds == class_labels).sum().item()
            correct_superclass += (superclass_preds == superclass_labels).sum().item()
            correct_group += (group_preds == group_labels).sum().item()

            class_preds_all.extend(class_preds.cpu().numpy())
            superclass_preds_all.extend(superclass_preds.cpu().numpy())
            group_preds_all.extend(group_preds.cpu().numpy())
            class_labels_all.extend(class_labels.cpu().numpy())
            superclass_labels_all.extend(superclass_labels.cpu().numpy())
            group_labels_all.extend(group_labels.cpu().numpy())

    print(f'Class Accuracy: {correct_class / total:.4f}, '
          f'Superclass Accuracy: {correct_superclass / total:.4f}, '
          f'Group Accuracy: {correct_group / total:.4f}')

    # Confusion Matrices plot
    plot_confusion_matrix(superclass_labels_all, superclass_preds_all, "Superclass Confusion Matrix", 20)
    plot_confusion_matrix(group_labels_all, group_preds_all, "Group Confusion Matrix", 9)


def plot_confusion_matrix(true_labels, pred_labels, title, num_classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

"""Weighted Sampling for Handling Class Imbalance"""

def get_sampler(subset):
    labels = [subset.dataset.superclass_labels[i] for i in subset.indices]
    num_classes = np.max(labels) + 1
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

"""Counting Trainable and Non-Trainable Parameters"""

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Total Trainable Parameters: {trainable_params}')
    print(f'Total Non-Trainable Parameters: {non_trainable_params}')

"""Main Function for Training and evaluating result for different Split ratio 70:30, 80:20, 90:10"""

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
])

    dataset = CIFAR100_Custom(root='./data', train=True, transform=transform, download=True)

    split_ratios = [0.7, 0.8, 0.9]
    for ratio in split_ratios:
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        sampler = get_sampler(train_set)
        train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2,drop_last=True)

        model = MultiTaskCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, num_epochs=20)
        evaluate_model(model, test_loader, criterion)
        count_parameters(model)

if __name__ == "__main__":
    main()

"""**Bonus Part** : Enhancing CNN Training with Severity-Based Misclassification Penalties"""

def compute_severity_penalty_matrix(dataset):
    num_classes = 100
    penalty_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            sup_i_name = dataset.get_superclass_name(dataset.superclass_labels[i])
            sup_j_name = dataset.get_superclass_name(dataset.superclass_labels[j])
            grp_i = dataset.superclass_mapping[sup_i_name]
            grp_j = dataset.superclass_mapping[sup_j_name]
            if sup_i_name == sup_j_name:
                penalty_matrix[i, j] = 1
            elif grp_i == grp_j:
                penalty_matrix[i, j] = 2
            else:
                penalty_matrix[i, j] = 3
    return torch.tensor(penalty_matrix, dtype=torch.float32)

# Severity Weighted Loss Function
def severity_weighted_loss(preds, labels, penalty_matrix, criterion):
    base_loss = criterion(preds, labels)
    _, preds_indices = torch.max(preds, 1)
    penalties = penalty_matrix[labels, preds_indices].to(preds.device)
    penalties[labels == preds_indices] = 1

    severity_loss = torch.mean(base_loss * penalties)
    return severity_loss

# Training Function with Severity-Based Loss
def train_model_severity(model, dataloader, criterion, optimizer, penalty_matrix, num_epochs=20, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, class_labels, superclass_labels, group_labels in dataloader:
            images, class_labels, superclass_labels, group_labels = (
                images.to(device), class_labels.to(device),
                superclass_labels.to(device), group_labels.to(device)
            )
            optimizer.zero_grad()
            class_output, superclass_output, group_output = model(images)
            class_loss = severity_weighted_loss(class_output, class_labels, penalty_matrix, criterion)
            superclass_loss = criterion(superclass_output, superclass_labels)
            group_loss = criterion(group_output, group_labels)
            loss = class_loss + 0.7 * superclass_loss + 0.5 * group_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

# Evaluation Function
def evaluate_model_severity(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    total_samples = 0
    correct_class, correct_superclass, correct_group = 0, 0, 0

    with torch.no_grad():
        for images, class_labels, superclass_labels, group_labels in dataloader:
            images, class_labels, superclass_labels, group_labels = (
                images.to(device), class_labels.to(device),
                superclass_labels.to(device), group_labels.to(device)
            )
            class_output, superclass_output, group_output = model(images)

            # Get predictions
            _, class_preds = torch.max(class_output, 1)
            _, superclass_preds = torch.max(superclass_output, 1)
            _, group_preds = torch.max(group_output, 1)

            # Update counts
            total_samples += class_labels.size(0)
            correct_class += (class_preds == class_labels).sum().item()
            correct_superclass += (superclass_preds == superclass_labels).sum().item()
            correct_group += (group_preds == group_labels).sum().item()

    print("\n=== Evaluation Results ===")
    print(f" Fine Class Accuracy: {correct_class / total_samples:.4f}")
    print(f" Superclass Accuracy: {correct_superclass / total_samples:.4f}")
    print(f" Group Accuracy: {correct_group / total_samples:.4f}")
    print("==========================\n")

# Main function with different train-test splits
def main_bonus():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
])

    dataset = CIFAR100_Custom(root='./data', train=True, transform=transform, download=True)
    penalty_matrix = compute_severity_penalty_matrix(dataset).to("cuda")

    split_ratios = [0.7, 0.8, 0.9]

    for ratio in split_ratios:
        print(f"\n=== Training with {int(ratio * 100)}:{int((1 - ratio) * 100)} Train-Test Split ===")

        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

        model = MultiTaskCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model_severity(model, train_loader, criterion, optimizer, penalty_matrix, num_epochs=20)

        # Evaluate Model after Training
        evaluate_model_severity(model, test_loader)

    print("\n=== Training & Evaluation Completed for All Splits ===")


if __name__ == "__main__":
    main_bonus()