import os
import pandas as pd

def parse_cliques(file_path, available_track_ids):
    cliques = []
    current_clique = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("%"):
                if len(current_clique) >= 2:
                    cliques.append(current_clique)
                current_clique = []
            else:
                parts = line.split("<SEP>")
                if len(parts) != 3:
                    continue
                track_id = parts[0].strip()
                if track_id in available_track_ids:
                    current_clique.append(track_id)

        if len(current_clique) >= 2:
            cliques.append(current_clique)

    return cliques


def generate_pairs(cliques):
    pairs = []
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                pairs.append((clique[i], clique[j], 1))  # positive pair
    return pairs


def generate_negative_pairs(positive_pairs, all_track_ids, num_negatives_per_positive=1):
    import random
    negatives = []
    positive_set = set((a, b) for a, b, _ in positive_pairs)

    for a, b, _ in positive_pairs:
        for _ in range(num_negatives_per_positive):
            c = random.choice(all_track_ids)
            while (a, c) in positive_set or (c, a) in positive_set or c == a:
                c = random.choice(all_track_ids)
            negatives.append((a, c, 0))

    return negatives
