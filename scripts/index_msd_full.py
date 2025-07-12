import os
import pickle

msd_root = r"e:\1 - Studia\Data Science\Magisterka\MilionSongSet"
output_file = "data/processed/msd_track_index_full.pkl"

track_index = {}

print("Indexing full MSD...")
for root, _, files in os.walk(msd_root):
    for file in files:
        if file.endswith(".h5"):
            track_id = file.replace(".h5", "")
            full_path = os.path.join(root, file)
            track_index[track_id] = full_path

print(f"Found {len(track_index)} tracks.")
with open(output_file, "wb") as f:
    pickle.dump(track_index, f)
print(f"Track index saved to {output_file}")