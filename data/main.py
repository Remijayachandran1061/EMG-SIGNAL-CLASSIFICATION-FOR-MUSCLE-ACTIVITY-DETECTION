import numpy as np
import pandas as pd
import joblib
import os

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

DATA_PATH = r"data/EMG-data2.csv"

label_map = {0: "REST", 1: "FLEX", 2: "GRIP"}

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

emg_cols = [
    "channel1","channel2","channel3","channel4",
    "channel5","channel6","channel7","channel8"
]

data = df[emg_cols].values

print("Total Samples:", len(data))

# ===============================
# FEATURE ENGINEERING
# ===============================
def extract_features(window):
    window = pd.DataFrame(window).rolling(3).mean().fillna(0).values

    rms = np.sqrt(np.mean(window**2, axis=0))
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    max_val = np.max(window, axis=0)
    min_val = np.min(window, axis=0)
    mav = np.mean(np.abs(window), axis=0)
    zcr = np.mean(np.diff(np.sign(window), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    return np.concatenate([rms, mean, std, max_val, min_val, mav, zcr, wl])

# ===============================
# PREDICTION
# ===============================
window_size = 80
predictions = []

for i in range(0, len(data) - window_size, window_size):
    window = data[i:i+window_size]

    features = extract_features(window)
    features = scaler.transform([features])

    pred = model.predict(features)[0]
    predictions.append(pred)

predictions = np.array(predictions)

# ===============================
# SAVE SAFELY (IMPORTANT FIX)
# ===============================
save_path = os.path.join(os.getcwd(), "predictions.npy")
np.save(save_path, predictions)

print("\n✔ Saved predictions at:", save_path)

# ===============================
# SUMMARY
# ===============================
unique, counts = np.unique(predictions, return_counts=True)

print("\nClass Distribution:")
for u, c in zip(unique, counts):
    print(label_map[u], ":", c)

print("\n✔ MAIN PIPELINE COMPLETE")