import numpy as np
from sklearn.linear_model import LinearRegression

# === Feature weights based on 1 / avg std from v5 analysis ===
FEATURE_WEIGHTS = np.array([
    1/17.22, 1/2.18, 1/0.16, 1/0.18, 1/0.10, 1/2.40,  # base (6)
    1, 1, 1/39.50, 1/0.82,                          # base (4 more)
    *[1/0.0001]*12,                                  # pitch slopes
    *[1/0.0045, 1/0.0323, 1/0.0330, 1/0.0115, 1/0.0193, 1/0.0098,
      1/0.0217, 1/0.0108, 1/0.0095, 1/0.0068, 1/0.0043, 1/0.0093]   # timbre slopes
], dtype=np.float32)

# Normalize weights
FEATURE_WEIGHTS /= FEATURE_WEIGHTS.max()


def extract_features_from_h5(h5):
    try:
        song = h5['analysis']['songs'][0]
        base = np.array([
            song['tempo'], song['key'], song['key_confidence'], song['mode'],
            song['mode_confidence'], song['loudness'], song['energy'], song['danceability'],
            song['duration'], song['time_signature']
        ], dtype=np.float32)

        pitches = h5['analysis']['segments_pitches'][:]
        timbre = h5['analysis']['segments_timbre'][:]
        if pitches.shape[0] < 2 or timbre.shape[0] < 2:
            raise ValueError("Not enough segments")

        pitches = (pitches - pitches.mean(axis=0)) / (pitches.std(axis=0) + 1e-6)
        timbre = (timbre - timbre.mean(axis=0)) / (timbre.std(axis=0) + 1e-6)
        time = np.arange(pitches.shape[0]).reshape(-1, 1)

        pitch_slopes = [LinearRegression().fit(time, pitches[:, i]).coef_[0] for i in range(12)]
        timbre_slopes = [LinearRegression().fit(time, timbre[:, i]).coef_[0] for i in range(12)]

        trend_features = np.concatenate([pitch_slopes, timbre_slopes], dtype=np.float32)
        full_features = np.concatenate([base, trend_features])

        # === Weight features ===
        if len(full_features) != len(FEATURE_WEIGHTS):
            raise ValueError("Feature length mismatch")
        full_features *= FEATURE_WEIGHTS

        return {
            'TrackID': song['track_id'].decode('utf-8') if isinstance(song['track_id'], bytes) else song['track_id'],
            'features': full_features
        }
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
