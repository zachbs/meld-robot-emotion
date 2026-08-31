import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report
import numpy as np

# ---------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------
FEATURE_PATH = "data/cached_features.pt"
TEST_PATH = "data/cached_features_test.pt"
VAL_PATH = "data/cached_features_dev.pt"
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 3e-4  # Adjusted for stable convergence with weight decay
NUM_CLASSES = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {DEVICE}")

# ---------------------------------------------------------
# 1. Custom PyTorch Dataset
# ---------------------------------------------------------
class MELDFeatureDataset(Dataset):
    def __init__(self, feature_path):
        print(f"Loading cached dataset from {feature_path}...")
        self.data = torch.load(feature_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['features'], sample['label']

# ---------------------------------------------------------
# 2. MLP Classifier Architecture
# ---------------------------------------------------------
class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.50),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# Helper: Compute Inverse Class Weights for Imbalanced Data
# ---------------------------------------------------------
def compute_class_weights(train_dataset, num_classes):
    labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    labels = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
    counts = np.bincount(labels, minlength=num_classes)
    
    total_samples = len(labels)
    # Square root smoothing prevents extreme multipliers > 5.0
    raw_weights = total_samples / (num_classes * np.maximum(counts, 1).astype(np.float32))
    smoothed_weights = np.sqrt(raw_weights)
    
    return torch.tensor(smoothed_weights, dtype=torch.float32)

# ---------------------------------------------------------
# 3. Main Training & Evaluation Routine
# ---------------------------------------------------------
def main():
    train_dataset = MELDFeatureDataset(FEATURE_PATH)
    val_dataset = MELDFeatureDataset(VAL_PATH)
    test_dataset = MELDFeatureDataset(TEST_PATH)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. Compute Class Weights on Training Data
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(DEVICE)
    print(f"Computed Class Weights: {class_weights.cpu().numpy().round(2)}")

    # 2. Setup Model, Loss, Optimizer, and LR Scheduler
    model = MultimodalEmotionClassifier(input_dim=1280, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    best_val_acc = 0.0

    print(f"\nStarting training on {train_size} samples, validating on {val_size} samples...")

    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=-1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = total_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # --- Validation Phase (Dev Set) ---
        model.eval()
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                preds = outputs.argmax(dim=-1)

                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = (correct_val / total_val) * 100.0
        val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        # --- Checkpointing: Save only on peak Dev F1 ---
        saved_str = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), "emotion_classifier_best.pth")
            saved_str = " -> [SAVED BEST]"

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | LR: {current_lr:.1e}{saved_str}")

    print(f"\nTraining Complete! Peak Dev Val Acc: {best_val_acc:.2f}% | Peak Dev Val Weighted F1: {best_val_f1:.2f}%")

    # ---------------------------------------------------------
    # 4. Final Evaluation on Dev Set & Official Test Set
    # ---------------------------------------------------------
    print("\nLoading best model checkpoint for evaluation...")
    model.load_state_dict(torch.load("emotion_classifier_best.pth"))
    model.eval()

    emotion_names = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]

    # --- Validation (Dev) Set Report ---
    dev_preds, dev_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            preds = outputs.argmax(dim=-1)
            dev_preds.extend(preds.cpu().numpy())
            dev_labels.extend(labels.numpy())

    print("\n=================== DEV SET REPORT ===================")
    print(classification_report(dev_labels, dev_preds, target_names=emotion_names, digits=4))

    # --- Official Test Set Report ---
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            preds = outputs.argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_acc = (np.array(test_preds) == np.array(test_labels)).mean() * 100.0
    test_f1 = f1_score(test_labels, test_preds, average='weighted') * 100.0

    print(f"\n=================== OFFICIAL TEST SET REPORT ({test_size} samples) ===================")
    print(f"Official Test Accuracy: {test_acc:.2f}% | Official Test Weighted F1: {test_f1:.2f}%\n")
    print(classification_report(test_labels, test_preds, target_names=emotion_names, digits=4))

if __name__ == "__main__":
    main()