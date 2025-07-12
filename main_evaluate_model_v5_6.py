import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from scripts.siamese_model_v5 import SiameseNetworkV5

# === CONFIG ===
TEST_SET_PATH = "data/processed/msd_test_set_v5_6.pkl"
TEST_PAIRS_PATH = "data/processed/test_pairs.csv"
MODEL_PATH = "models/siamese_model_v5_6_.pth"
MAX_QUERIES = None  # set to integer for subset eval (e.g., 100)

# === Load test set and model ===
print("Loading test set and model...")
with open(TEST_SET_PATH, "rb") as f:
    test_set = pickle.load(f)
test_pairs = pd.read_csv(TEST_PAIRS_PATH)

# === Prepare feature map ===
feature_map = {}
for item in test_set:
    features = item.get("features", None)
    if isinstance(features, np.ndarray) and len(features) == 34:
        feature_map[item["TrackID"]] = features

feature_counts = pd.Series([len(f) for f in feature_map.values()]).value_counts()
print("features\n", feature_counts)

# Filter valid test pairs
valid_pairs = test_pairs[
    test_pairs['TrackID_A'].isin(feature_map) &
    test_pairs['TrackID_B'].isin(feature_map)
]
print(f"Filtered test pairs: {len(valid_pairs)} (removed {len(test_pairs) - len(valid_pairs)} invalid rows)")

if valid_pairs.empty:
    print(" No valid test pairs found.")
    exit()

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetworkV5(input_dim=34)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Generate embeddings ===
print("Embedding test tracks...")
embeddings = {}
with torch.no_grad():
    for tid, feat in tqdm(feature_map.items(), desc="Embedding tracks"):
        tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        emb = model.embedding(tensor).squeeze().cpu().numpy()
        embeddings[tid] = emb

# === MAP@10 Evaluation ===
def mean_average_precision_at_k(pairs_df, embeddings, k=10, max_queries=None):
    from collections import defaultdict

    truth = defaultdict(set)
    for _, row in pairs_df.iterrows():
        truth[row["TrackID_A"]].add(row["TrackID_B"])

    queries = list(truth.keys())
    if max_queries:
        queries = queries[:max_queries]

    ap_list = []

    print(f"Evaluating {len(queries)} queries...")
    for query in tqdm(queries, desc="Evaluating MAP@10"):
        if query not in embeddings:
            continue
        query_emb = embeddings[query]
        dists = []
        for tid, emb in embeddings.items():
            if tid == query:
                continue
            dist = 1 - F.cosine_similarity(torch.tensor(query_emb), torch.tensor(emb), dim=0).item()
            dists.append((tid, dist))
        dists.sort(key=lambda x: x[1])
        top_k = [tid for tid, _ in dists[:k]]

        hits, ap = 0, 0.0
        for i, tid in enumerate(top_k):
            if tid in truth[query]:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap_list.append(ap / hits)

    return np.mean(ap_list) if ap_list else 0.0

# === Run Evaluation ===
map10 = mean_average_precision_at_k(valid_pairs, embeddings, k=10, max_queries=MAX_QUERIES)
print(f"\n MAP@10 on test set (max_queries={MAX_QUERIES} queries): {map10:.4f}")

