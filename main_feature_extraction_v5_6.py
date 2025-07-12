import pickle
from tqdm import tqdm
import numpy as np
from scripts.utils import load_h5_file
from scripts.extract_features_v5_6 import extract_features_from_h5

# === CONFIG ===
TRACK_INDEX_PATH = "data/processed/msd_track_index_full.pkl"
OUTPUT_TRAIN_SET = "data/processed/msd_train_set_v5_6.pkl"
OUTPUT_TEST_SET = "data/processed/msd_test_set_v5_6.pkl"
SHS_TRAIN_PATH = "data/shs_dataset_train.txt"
SHS_TEST_PATH = "data/shs_dataset_test.txt"

def parse_clique_file(path):
    cliques = []
    current_clique = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%'):
                if current_clique:
                    cliques.append(current_clique)
                current_clique = []
            elif line and not line.startswith('#'):
                parts = line.split('<SEP>')
                if len(parts) == 3:
                    track_id = parts[0]
                    current_clique.append(track_id)
    if current_clique:
        cliques.append(current_clique)
    return cliques

def get_unique_track_ids(cliques):
    track_ids = set()
    for clique in cliques:
        track_ids.update(clique)
    return track_ids

def build_dataset(track_ids, track_index):
    dataset = []
    skipped_count = 0
    for track_id in tqdm(track_ids, desc="Extracting features"):
        path = track_index.get(track_id)
        if not path:
            skipped_count += 1
            continue
        try:
            h5 = load_h5_file(path)
            result = extract_features_from_h5(h5)
            if result is None or len(result['features']) != 34:
                skipped_count += 1
                continue
            dataset.append({"TrackID": track_id, "features": result['features']})
        except Exception:
            skipped_count += 1
            continue
    return dataset, skipped_count

# === MAIN ===
print("Loading track index...")
with open(TRACK_INDEX_PATH, "rb") as f:
    track_index = pickle.load(f)

print("Parsing SHS train/test cliques...")
train_cliques = parse_clique_file(SHS_TRAIN_PATH)
test_cliques = parse_clique_file(SHS_TEST_PATH)

train_track_ids = get_unique_track_ids(train_cliques)
test_track_ids = get_unique_track_ids(test_cliques)

print(f"Train track IDs: {len(train_track_ids)}, Test track IDs: {len(test_track_ids)}")

# Filter valid MSD IDs
valid_train_ids = [tid for tid in train_track_ids if tid in track_index]
valid_test_ids = [tid for tid in test_track_ids if tid in track_index]

print(f"Valid in MSD - Train: {len(valid_train_ids)}, Test: {len(valid_test_ids)}")

print("Extracting features for training set...")
train_set, train_skipped = build_dataset(valid_train_ids, track_index)
print("Extracting features for test set...")
test_set, test_skipped = build_dataset(valid_test_ids, track_index)

with open(OUTPUT_TRAIN_SET, "wb") as f:
    pickle.dump(train_set, f)
with open(OUTPUT_TEST_SET, "wb") as f:
    pickle.dump(test_set, f)

print("\n Feature extraction complete.")
print(f"Saved {len(train_set)} training samples, skipped {train_skipped} invalid tracks.")
print(f"Saved {len(test_set)} test samples, skipped {test_skipped} invalid tracks.")