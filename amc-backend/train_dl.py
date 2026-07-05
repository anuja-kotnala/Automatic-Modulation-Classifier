import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)

from amc.utils.visualization import plot_confusion_matrix

# ==========================================================
# 1. Custom Datasets for PyTorch
# ==========================================================

class IQDataset(Dataset):
    """Dataset for raw IQ signals (2 x 1024)."""
    def __init__(self, metadata_df: pd.DataFrame, dataset_dir: str):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.dataset_dir, row["file_path"])
        
        # Load complex IQ signal: shape (1024,)
        signal = np.load(file_path)
        
        # Stack Real and Imaginary parts to get shape (2, 1024)
        iq_data = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)
        label = int(row["label"])
        
        return torch.tensor(iq_data), torch.tensor(label, dtype=torch.long)

class FeatureDataset(Dataset):
    """Dataset for tabular features (e.g. 39 extracted metrics)."""
    def __init__(self, features_df: pd.DataFrame, feature_cols: list):
        self.metadata = features_df.reset_index(drop=True)
        self.X = self.metadata[feature_cols].values.astype(np.float32)
        self.y = self.metadata["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # We reshape to (1, num_features) to allow 1D convolution operations
        x_data = self.X[idx].reshape(1, -1)
        return torch.tensor(x_data), torch.tensor(self.y[idx], dtype=torch.long)

class SpectrogramDataset(Dataset):
    """Dataset for 2D Spectrogram representation of raw IQ signals."""
    def __init__(self, metadata_df: pd.DataFrame, dataset_dir: str, nperseg: int = 64, noverlap: int = 32):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.dataset_dir, row["file_path"])
        signal = np.load(file_path)
        
        # Compute spectrogram for Real and Imaginary parts separately to preserve phase/quadrature alignment
        from scipy.signal import spectrogram
        _, _, Sxx_real = spectrogram(
            signal.real,
            fs=1.0,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False
        )
        _, _, Sxx_imag = spectrogram(
            signal.imag,
            fs=1.0,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False
        )
        
        # Center frequencies
        Sxx_real = np.fft.fftshift(Sxx_real, axes=0)
        Sxx_imag = np.fft.fftshift(Sxx_imag, axes=0)
        # Convert to dB
        Sxx_real_db = 10 * np.log10(Sxx_real + 1e-12).astype(np.float32)
        Sxx_imag_db = 10 * np.log10(Sxx_imag + 1e-12).astype(np.float32)
        
        # Pad or truncate width dimension to exactly 31 columns
        def pad_or_truncate(S):
            if S.shape[1] < 31:
                pad_width = 31 - S.shape[1]
                S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
            elif S.shape[1] > 31:
                S = S[:, :31]
            return S
            
        Sxx_real_db = pad_or_truncate(Sxx_real_db)
        Sxx_imag_db = pad_or_truncate(Sxx_imag_db)
        
        # Shape: (2, nperseg, num_times)
        spectrogram_data = np.stack([Sxx_real_db, Sxx_imag_db], axis=0)
        label = int(row["label"])
        
        return torch.tensor(spectrogram_data), torch.tensor(label, dtype=torch.long)

# ==========================================================
# 2. PyTorch Deep Learning Model Architectures
# ==========================================================

class CNN1D(nn.Module):
    """1D CNN architecture."""
    def __init__(self, in_channels: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, 256, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        return self.fc(features)

class CNNLSTM(nn.Module):
    """CNN + LSTM architecture."""
    def __init__(self, in_channels: int, sequence_length: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4), # sequence_length / 4
            nn.Dropout1d(p=0.4)  # Spatial Dropout
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        features = self.conv(x)  # shape: (batch, 64, reduced_len)
        features = features.transpose(1, 2)  # shape: (batch, reduced_len, 64)
        lstm_out, (hn, cn) = self.lstm(features)  # hn shape: (2, batch, 64)
        
        # Concatenate the final forward hidden state and backward hidden state
        # hn[0] is the forward final state, hn[1] is the backward final state
        hn_cat = torch.cat((hn[0], hn[1]), dim=1)  # shape: (batch, 128)
        
        return self.fc(hn_cat)

class CNN2D(nn.Module):
    """2D CNN architecture operating on spectrograms (image-like representations)."""
    def __init__(self, in_channels: int, height: int, width: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (batch_size, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, in_channels, height, width)
        features = self.conv(x).squeeze(-1).squeeze(-1)
        return self.fc(features)

# ==========================================================
# 3. Training Callbacks & Early Stopping
# ==========================================================

class EarlyStopping:
    """Terminates training if validation loss does not improve."""
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# ==========================================================
# 4. Modulated Helper Functions
# ==========================================================

def get_model(model_name: str, in_channels: int, seq_len: int, num_classes: int) -> nn.Module:
    m_name = model_name.lower().strip()
    if m_name == "cnn1d":
        return CNN1D(in_channels, seq_len, num_classes)
    elif m_name == "cnnlstm":
        return CNNLSTM(in_channels, seq_len, num_classes)
    elif m_name == "cnn2d":
        nperseg = 64
        noverlap = 32
        height = nperseg
        width = (seq_len - noverlap) // (nperseg - noverlap)
        return CNN2D(in_channels, height, width, num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def load_and_preprocess_data(input_type: str, model_name: str, batch_size: int, dataset_dir: str = "dataset"):
    """Loads datasets, processes labels, performs Train(70%)/Val(15%)/Test(15%) splits, and returns DataLoader instances."""
    if input_type == "raw":
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Run generate_dataset.py first to create raw signal files.")
        
        df_meta = pd.read_csv(metadata_path)
        label_encoder = LabelEncoder()
        df_meta["label"] = label_encoder.fit_transform(df_meta["modulation"])
        classes = label_encoder.classes_
        
        # Split: 70% Train, 15% Val, 15% Test
        train_val_df, test_df = train_test_split(df_meta, test_size=0.15, random_state=42, stratify=df_meta["label"])
        train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42, stratify=train_val_df["label"])
        
        if model_name.lower().strip() == "cnn2d":
            train_dataset = SpectrogramDataset(train_df, dataset_dir)
            val_dataset = SpectrogramDataset(val_df, dataset_dir)
            test_dataset = SpectrogramDataset(test_df, dataset_dir)
            in_channels = 2
        else:
            train_dataset = IQDataset(train_df, dataset_dir)
            val_dataset = IQDataset(val_df, dataset_dir)
            test_dataset = IQDataset(test_df, dataset_dir)
            in_channels = 2
            
        seq_len = 1024
    else:
        features_path = os.path.join(dataset_dir, "features.csv")
        if not os.path.exists(features_path):
            raise FileNotFoundError("Run extract_features.py first to compile features.csv.")
            
        df_feats = pd.read_csv(features_path)
        label_encoder = LabelEncoder()
        df_feats["label"] = label_encoder.fit_transform(df_feats["modulation"])
        classes = label_encoder.classes_
        
        meta_cols = ["file_path", "modulation", "snr_db", "label"]
        feature_cols = [col for col in df_feats.columns if col not in meta_cols]
        
        # Standardize features
        scaler = StandardScaler()
        df_feats[feature_cols] = scaler.fit_transform(df_feats[feature_cols].values)
        
        # Split: 70% Train, 15% Val, 15% Test
        train_val_df, test_df = train_test_split(df_feats, test_size=0.15, random_state=42, stratify=df_feats["label"])
        train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42, stratify=train_val_df["label"])
        
        train_dataset = FeatureDataset(train_df, feature_cols)
        val_dataset = FeatureDataset(val_df, feature_cols)
        test_dataset = FeatureDataset(test_df, feature_cols)
        
        in_channels = 1
        seq_len = len(feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, in_channels, seq_len, classes

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Executes a single training epoch."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == targets).sum().item()
        total_train += targets.size(0)

    epoch_train_loss = train_loss / total_train
    epoch_train_acc = train_correct / total_train
    return epoch_train_loss, epoch_train_acc

def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates the model on the provided loader and returns loss, accuracy, targets, predictions, and probabilities."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == targets).sum().item()
            total_val += targets.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = val_correct / total_val
    return epoch_val_loss, epoch_val_acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)

def train_model(model_name: str, input_type: str, epochs: int, batch_size: int, lr: float, device: torch.device,
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                in_channels: int, seq_len: int, classes: list, models_dir: str, eval_dir: str):
    """Orchestrates the entire training pipeline for a given model architecture."""
    import time
    # Initialize TensorBoard Logging
    writer = SummaryWriter(log_dir=f"runs/amc_dl_{input_type}_{model_name}")

    num_classes = len(classes)
    model = get_model(model_name, in_channels, seq_len, num_classes).to(device)

    # Delete any pre-existing checkpoint to prevent stale evaluations/learning curve mismatches
    checkpoint_path = os.path.join(models_dir, f"best_{model_name}_{input_type}.pth")
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"Removed pre-existing checkpoint file: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not remove pre-existing checkpoint: {e}")

    # Set stabilization hyperparameters: lower learning rate for cnn2d, custom optimizer config, weight decay and patience
    current_lr = 1e-4 if model_name.lower().strip() == "cnn2d" else lr
    weight_decay = 1e-4 if model_name.lower().strip() == "cnnlstm" else 0.0
    patience = 3 if model_name.lower().strip() == "cnnlstm" else 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining model {model_name.upper()} on {input_type.upper()} signal representations...")

    start_train_time = time.perf_counter()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_train_loss, epoch_train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_val_loss, epoch_val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_end = time.perf_counter()
        epoch_times.append(epoch_end - epoch_start)

        # Update Scheduler and History
        scheduler.step(epoch_val_loss)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", epoch_val_acc, epoch)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Model Checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(models_dir, f"best_{model_name}_{input_type}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'classes': classes
            }, checkpoint_path)
            print(f"--> Saved best checkpoint to {checkpoint_path} (Val Acc: {epoch_val_acc:.4f})")

        # Early Stopping check
        if early_stopping(epoch_val_loss):
            print("Early stopping triggered. Terminating training loop.")
            break

    writer.close()

    total_train_time = time.perf_counter() - start_train_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    # Load best model checkpoint for evaluation on test set
    checkpoint_path = os.path.join(models_dir, f"best_{model_name}_{input_type}.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} for final testing.")

    # Model size on disk (.pth file size)
    model_size_mb = 0.0
    if os.path.exists(checkpoint_path):
        model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # Final Evaluation on the Test Set & Inference time measurement
    test_start = time.perf_counter()
    test_loss, test_acc, test_targets, test_preds, test_probs = evaluate_model(model, test_loader, criterion, device)
    test_end = time.perf_counter()

    total_inference_time = test_end - test_start
    num_samples = len(test_loader.dataset)
    avg_inference_time_per_sample_ms = (total_inference_time / num_samples) * 1000.0 if num_samples > 0 else 0.0

    # Accuracy vs SNR calculation
    snr_accuracies = {}
    if hasattr(test_loader.dataset, 'metadata') and "snr_db" in test_loader.dataset.metadata.columns:
        snr_list = test_loader.dataset.metadata["snr_db"].values
        for unique_snr in np.unique(snr_list):
            indices = (snr_list == unique_snr)
            if np.sum(indices) > 0:
                acc = np.mean(test_preds[indices] == test_targets[indices])
                snr_accuracies[int(float(unique_snr))] = float(acc)

    # Plot single model loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue", marker='o')
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="red", linestyle="--", marker='o')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training & Validation Loss ({model_name.upper()})")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history["train_acc"], label="Train Acc", color="blue", marker='o')
    ax2.plot(epochs_range, history["val_acc"], label="Val Acc", color="red", linestyle="--", marker='o')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Training & Validation Accuracy ({model_name.upper()})")
    ax2.legend()
    ax2.grid(True)

    fig.savefig(os.path.join(eval_dir, f"{model_name}_{input_type}_learning_curves.png"))
    plt.close(fig)

    # Plot Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    fig_cm = plot_confusion_matrix(cm, classes=list(classes), title=f"{model_name.upper()} ({input_type.upper()}) Confusion Matrix")
    fig_cm.savefig(os.path.join(eval_dir, f"{model_name}_{input_type}_confusion_matrix.png"))
    plt.close(fig_cm)

    # Classification Report
    print(f"\nFinal Test Classification Report ({model_name.upper()}):")
    print(classification_report(test_targets, test_preds, target_names=classes))

    # Calculate additional metrics
    prec = precision_score(test_targets, test_preds, average="macro")
    rec = recall_score(test_targets, test_preds, average="macro")
    f1 = f1_score(test_targets, test_preds, average="macro")
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "params_count": params_count,
        "history": history,
        "targets": test_targets,
        "preds": test_preds,
        "probs": test_probs,
        "training_time": total_train_time,
        "avg_epoch_time": avg_epoch_time,
        "model_size_mb": model_size_mb,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "avg_inference_time_per_sample_ms": avg_inference_time_per_sample_ms,
        "snr_accuracies": snr_accuracies
    }
    return metrics

def plot_comparisons(results: dict, eval_dir: str, classes: list):
    """Generates comparison plots across all evaluated models."""
    # 1. Compare Validation Accuracy Curves
    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        val_acc = metrics["history"]["val_acc"]
        if len(val_acc) == 1:
            plt.scatter(1, val_acc[0], s=100, label=f"{model_name.upper()}")
            plt.axhline(y=val_acc[0], linestyle="--", alpha=0.5)
        else:
            plt.plot(range(1, len(val_acc) + 1), val_acc, label=f"{model_name.upper()}", linewidth=2, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "dl_accuracy_comparison.png"))
    plt.close()

    # 2. Compare Macro-Average ROC Curves
    plt.figure(figsize=(10, 8))
    n_classes = len(classes)
    
    for model_name, metrics in results.items():
        y_test = metrics["targets"]
        y_prob = metrics["probs"]
        
        # Compute macro-average ROC curve
        fpr_dict = dict()
        tpr_dict = dict()
        
        for i in range(n_classes):
            y_test_bin = (y_test == i).astype(int)
            y_prob_class = y_prob[:, i]
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin, y_prob_class)
            
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f"{model_name.upper()} (Macro AUC = {macro_auc:.3f})", linewidth=1.5)

    plt.plot([0, 1], [0, 1], color='#666666', linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Average ROC Curves Comparison (Deep Learning)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "dl_roc_comparison.png"))
    plt.close()

# ==========================================================
# 5. Main Training Pipeline Entrypoint
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="AMC PyTorch Deep Learning Training Pipeline")
    parser.add_argument("--input_type", type=str, choices=["raw", "features"], default="raw", help="Training input type")
    parser.add_argument("--model", type=str, choices=["cnn1d", "cnnlstm", "cnn2d"], default="cnn1d", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--compare_all", action="store_true", help="Run training and comparison across all models")
    args = parser.parse_args()

    models_dir = os.path.join("results", "models", "dl")
    eval_dir = os.path.join("results", "dl_eval")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Deep Learning Pipeline using: {device}")

    # Enforce constraints for CNN2D
    if not args.compare_all and args.model.lower().strip() == "cnn2d":
        if args.input_type != "raw":
            raise ValueError("CNN2D architecture requires raw IQ signals to generate spectrogram representations. It cannot be used with '--input_type features'.")

    # Determine models to run
    if args.compare_all:
        if args.input_type == "features":
            models_to_run = ["cnn1d", "cnnlstm"]
            print("Running in comparison mode. Warning: 'cnn2d' will be skipped since '--input_type features' is selected.")
        else:
            models_to_run = ["cnn1d", "cnnlstm", "cnn2d"]
    else:
        models_to_run = [args.model]

    comparison_results = {}
    classes = None

    for m_name in models_to_run:
        print(f"\n" + "="*50)
        print(f"Starting Training for Model: {m_name.upper()}")
        print("="*50)

        # Load appropriate data loaders for this specific model configuration
        train_loader, val_loader, test_loader, in_channels, seq_len, classes = load_and_preprocess_data(
            input_type=args.input_type,
            model_name=m_name,
            batch_size=args.batch_size
        )

        metrics = train_model(
            model_name=m_name,
            input_type=args.input_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            in_channels=in_channels,
            seq_len=seq_len,
            classes=classes,
            models_dir=models_dir,
            eval_dir=eval_dir
        )
        comparison_results[m_name] = metrics

    if args.compare_all:
        # Create comparison summary table
        records = []
        for m_name, metrics in comparison_results.items():
            record = {
                "Model": m_name.upper(),
                "Test Accuracy": metrics["test_acc"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "Parameters": metrics["params_count"],
                "Training Time (s)": metrics["training_time"],
                "Avg Epoch Time (s)": metrics["avg_epoch_time"],
                "Model Size (MB)": metrics["model_size_mb"],
                "Best Val Acc": metrics["best_val_acc"],
                "Best Val Loss": metrics["best_val_loss"],
                "Avg Inference Time/Sample (ms)": metrics["avg_inference_time_per_sample_ms"]
            }
            # Add SNR accuracies as separate columns if available
            for snr_val, acc in sorted(metrics["snr_accuracies"].items()):
                record[f"Acc_SNR_{snr_val}dB"] = acc
            records.append(record)
        df_summary = pd.DataFrame(records)
        summary_path = os.path.join(eval_dir, "dl_model_performance_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\nSaved comparison performance summary to {summary_path}")
        print(df_summary.to_string(index=False))

        # Generate comparative charts
        plot_comparisons(comparison_results, eval_dir, classes)
        print(f"Generated comparison plots under: {eval_dir}")

if __name__ == "__main__":
    main()
