import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np
from scripts.siamese_model_v5 import SiameseNetworkV5
from tqdm import tqdm
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === CONFIG ===
TRAIN_SET_PATH = "data/processed/msd_train_set_v5_6.pkl"
TRAIN_PAIRS_PATH = "data/processed/train_pairs.csv"
MODEL_SAVE_PATH = "models/siamese_model_v5_6.pth"
NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
set_seed(5)

# === Dataset ===
class TrackPairDataset(Dataset):
    def __init__(self, pairs_df, feature_dict):
        self.pairs_df = pairs_df
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        feat1 = self.feature_dict[row['TrackID_A']]
        feat2 = self.feature_dict[row['TrackID_B']]

        return (
            torch.tensor(feat1, dtype=torch.float32),
            torch.tensor(feat2, dtype=torch.float32),
            torch.tensor(row['Label'], dtype=torch.float32)
        )

# === Contrastive Loss ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# === Load Data ===
print("Loading feature set and training pairs...")
with open(TRAIN_SET_PATH, "rb") as f:
    train_set = pickle.load(f)
train_pairs = pd.read_csv(TRAIN_PAIRS_PATH)

# === Prepare Feature Map ===
feature_map = {}
for item in train_set:
    features = item.get("features", None)
    if isinstance(features, np.ndarray) and len(features) == 34:
        feature_map[item["TrackID"]] = features

feature_counts = pd.Series([len(f) for f in feature_map.values()]).value_counts()
print("features\n", feature_counts)

valid_pairs = train_pairs[
    train_pairs["TrackID_A"].isin(feature_map) &
    train_pairs["TrackID_B"].isin(feature_map)
]
print(f"Filtered training pairs: {len(valid_pairs)} (removed {len(train_pairs) - len(valid_pairs)} invalid rows)")

if valid_pairs.empty:
    print(" No valid training data. Check your feature extraction.")
    exit()

# === Training Setup ===
dataset = TrackPairDataset(valid_pairs, feature_map)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SiameseNetworkV5(
    input_dim=34,
    hidden_dims=[128, 64],     # or try [64], or [512, 256, 128], etc.
    embedding_dim=256,          # or 64, 256, etc.
    dropout=0.2                 # optional: tweak regularization
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", ncols=100)
    for f1, f2, label in loop:
        f1, f2, label = f1.to(device), f2.to(device), label.to(device)

        out1, out2 = model(f1, f2)
        loss = criterion(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

# === Save Model ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f" Model saved to {MODEL_SAVE_PATH}")
