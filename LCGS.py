import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# 配置
# =========================
DATA_DIR = "dataset_loc_bus12368_csv"
LABEL_PATH = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")

BATCH_SIZE = 128
EPOCHS = 50
LR = 5e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
NUM_WORKERS = 0
RANDOM_SEED = 41

TIME_START = 0.015
TIME_END   = 0.085

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "best_lstm_loc_bus12368.pth"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 多分类标签映射
BUS_TO_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    6: 3,
    8: 4
}

LABEL_TO_NAME = {
    0: "Bus1",
    1: "Bus2",
    2: "Bus3",
    3: "Bus6",
    4: "Bus8"
}

TARGET_NAMES = ["Bus1", "Bus2", "Bus3", "Bus6", "Bus8"]
NUM_CLASSES = 5


# =========================
# 数据集
# =========================
class OscillationDataset(Dataset):
    """
    输入样本原始列（来自 MATLAB 导出的 dataset_loc_bus12368_csv）：
    time,
    Bus1_A, Bus1_B, Bus1_C,
    Bus2_A, Bus2_B, Bus2_C,
    Bus3_A, Bus3_B, Bus3_C,
    Bus5_A, Bus5_B, Bus5_C,
    Bus6_A, Bus6_B, Bus6_C,
    Bus8_A, Bus8_B, Bus8_C,
    Bus9_A, Bus9_B, Bus9_C

    共 7 个节点 × 3 相 = 21 通道
    """

    def __init__(self, labels_df, sample_dir, time_start=0.015, time_end=0.085, add_diff_features=True):
        self.labels_df = labels_df.reset_index(drop=True)
        self.sample_dir = sample_dir
        self.time_start = time_start
        self.time_end = time_end
        self.add_diff_features = add_diff_features

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        file_name = row["file_name"]
        src_bus = int(row["src_bus"])

        label = BUS_TO_LABEL[src_bus]

        file_path = os.path.join(self.sample_dir, file_name)
        df = pd.read_csv(file_path)

        # ===== 截取时间窗 =====
        df = df[(df["time"] >= self.time_start) & (df["time"] <= self.time_end)].reset_index(drop=True)

        # ===== 原始特征 =====
        feat_cols = [c for c in df.columns if c != "time"]
        X = df[feat_cols].values.astype(np.float32)   # (T, 21)

        # ===== 加空间差分特征 =====
        # 当前默认列顺序：
        # Bus1(0:3), Bus2(3:6), Bus3(6:9), Bus5(9:12),
        # Bus6(12:15), Bus8(15:18), Bus9(18:21)
        if self.add_diff_features:
            bus1 = X[:, 0:3]
            bus2 = X[:, 3:6]
            bus3 = X[:, 6:9]
            bus5 = X[:, 9:12]
            bus6 = X[:, 12:15]
            bus8 = X[:, 15:18]
            bus9 = X[:, 18:21]

            # 选一些有物理意义的差分特征
            diff_18 = bus1 - bus8
            diff_28 = bus2 - bus8
            diff_38 = bus3 - bus8
            diff_68 = bus6 - bus8
            diff_58 = bus5 - bus8
            diff_98 = bus9 - bus8

            diff_12 = bus1 - bus2
            diff_23 = bus2 - bus3
            diff_36 = bus3 - bus6

            X = np.concatenate([
                X,
                diff_18, diff_28, diff_38, diff_68, diff_58, diff_98,
                diff_12, diff_23, diff_36
            ], axis=1)
            # 原始 21 通道 + 9组差分*3 = 48 通道

        # ===== 按通道标准化 =====
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mean) / std

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return X, y


# =========================
# LSTM 模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, num_classes=5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)      # (B, T, H)

        # 时间平均池化
        out = out.mean(dim=1)      # (B, H)

        out = self.dropout(out)
        logits = self.fc(out)      # (B, 5)
        return logits


# =========================
# 训练一个 epoch
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# =========================
# 评估
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * X.size(0)
            all_labels.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro        = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_weighted    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_weighted        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    return {
        "loss": val_loss,
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "cm": cm,
        "labels": all_labels,
        "preds": all_preds
    }


# =========================
# 主流程
# =========================
def main():
    print(f"Using device: {DEVICE}")

    labels_df = pd.read_csv(LABEL_PATH)

    # 只保留有效类别
    valid_buses = sorted(BUS_TO_LABEL.keys())
    labels_df = labels_df[labels_df["src_bus"].isin(valid_buses)].reset_index(drop=True)

    print("\n===== Full Dataset Distribution =====")
    print(labels_df["src_bus"].value_counts().sort_index())

    # ===== 分层划分 =====
    train_df, val_df = train_test_split(
        labels_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels_df["src_bus"]
    )

    print("\n===== Train Distribution =====")
    print(train_df["src_bus"].value_counts().sort_index())

    print("\n===== Val Distribution =====")
    print(val_df["src_bus"].value_counts().sort_index())

    # ===== Dataset =====
    train_dataset = OscillationDataset(
        labels_df=train_df,
        sample_dir=SAMPLE_DIR,
        time_start=TIME_START,
        time_end=TIME_END,
        add_diff_features=True
    )

    val_dataset = OscillationDataset(
        labels_df=val_df,
        sample_dir=SAMPLE_DIR,
        time_start=TIME_START,
        time_end=TIME_END,
        add_diff_features=True
    )

    sample_X, sample_y = train_dataset[0]
    T, C = sample_X.shape

    print(f"\nInput shape after preprocessing: T={T}, C={C}")
    print(f"Sample label: {sample_y.item()} -> {LABEL_TO_NAME[sample_y.item()]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ===== 类别权重 =====
    train_counts = train_df["src_bus"].value_counts()
    class_weights = []
    for bus in [1, 2, 3, 6, 8]:
        n = train_counts.get(bus, 0)
        class_weights.append(1.0 / max(n, 1))

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(DEVICE)

    print("\nClass weights:", class_weights.detach().cpu().numpy())

    # ===== 模型 =====
    model = LSTMClassifier(
        input_size=C,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -1.0

    print("\n===== Start Training =====\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val F1(macro): {val_metrics['f1_macro']:.4f}"
        )

        if epoch % 10 == 0:
            print("\n===== Detailed Evaluation =====")
            print(f"Accuracy           : {val_metrics['acc']:.4f}")
            print(f"Precision (macro)  : {val_metrics['precision_macro']:.4f}")
            print(f"Recall (macro)     : {val_metrics['recall_macro']:.4f}")
            print(f"F1 Score (macro)   : {val_metrics['f1_macro']:.4f}")
            print(f"Precision(weighted): {val_metrics['precision_weighted']:.4f}")
            print(f"Recall(weighted)   : {val_metrics['recall_weighted']:.4f}")
            print(f"F1 Score(weighted) : {val_metrics['f1_weighted']:.4f}")

            print("\nConfusion Matrix:")
            print(val_metrics["cm"])

            print("\nClassification Report:")
            print(classification_report(
                val_metrics["labels"],
                val_metrics["preds"],
                labels=list(range(NUM_CLASSES)),
                target_names=TARGET_NAMES,
                digits=4,
                zero_division=0
            ))
            print("===============================\n")

        # 保存 best model（按 macro-F1）
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1_macro": best_val_f1,
                "input_size": C,
                "time_window": [TIME_START, TIME_END],
                "target_names": TARGET_NAMES
            }, BEST_MODEL_PATH)
            print(f"[*] Best model saved at epoch {epoch}, macro-F1 = {best_val_f1:.4f}")

    print("\nTraining finished.")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()