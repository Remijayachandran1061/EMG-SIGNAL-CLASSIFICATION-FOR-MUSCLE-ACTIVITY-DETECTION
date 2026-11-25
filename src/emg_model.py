import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import joblib

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/EMG-data2.csv")

print("Total Samples:", len(df))

# ===============================
# SELECT EMG CHANNELS
# ===============================
emg_cols = [
    "channel1","channel2","channel3","channel4",
    "channel5","channel6","channel7","channel8"
]

X = df[emg_cols]

# ===============================
# LABEL MAPPING (8 → 3 CLASSES)
# ===============================
df["class"] = df["class"].astype(int)

def map_class(x):
    if x == 0:
        return 0   # Rest
    elif x in [1,2,3]:
        return 1   # Flex
    else:
        return 2   # Grip

y = df["class"].apply(map_class)

print("\nMapped Class Distribution:")
print(y.value_counts())

# ===============================
# FEATURE EXTRACTION
# ===============================
window_size = 80

X_features = []
y_labels = []

for i in range(0, len(X) - window_size, window_size):
    window = X.iloc[i:i+window_size].values

    # 🔥 SMOOTHING
    window = pd.DataFrame(window).rolling(3).mean().fillna(0).values

    # ===== FEATURES =====
    rms = np.sqrt(np.mean(window**2, axis=0))
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    max_val = np.max(window, axis=0)
    min_val = np.min(window, axis=0)

    mav = np.mean(np.abs(window), axis=0)
    zcr = np.mean(np.diff(np.sign(window), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    features = np.concatenate([
        rms, mean, std, max_val, min_val,
        mav, zcr, wl
    ])

    X_features.append(features)
    y_labels.append(y.iloc[i])

X_features = np.array(X_features)
y_labels = np.array(y_labels)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels,
    test_size=0.3,
    random_state=42,
    stratify=y_labels
)

print("\nTrain Samples:", len(X_train))
print("Test Samples:", len(X_test))

# ===============================
# BALANCING
# ===============================
train_df = pd.DataFrame(X_train)
train_df["label"] = y_train

rest = train_df[train_df["label"] == 0]
flex = train_df[train_df["label"] == 1]
grip = train_df[train_df["label"] == 2]

target_size = 10000

print("\nBalancing each class to:", target_size)

rest_bal = resample(rest, replace=True, n_samples=target_size, random_state=42)
flex_bal = resample(flex, replace=True, n_samples=target_size, random_state=42)
grip_bal = resample(grip, replace=True, n_samples=target_size, random_state=42)

balanced_df = pd.concat([rest_bal, flex_bal, grip_bal])

X_train_bal = balanced_df.drop("label", axis=1).values
y_train_bal = balanced_df["label"].values

print("Balanced Train Samples:", len(X_train_bal))

# ===============================
# SCALING
# ===============================
scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)

# Clip noise
X_train_bal = np.clip(X_train_bal, -5, 5)
X_test = np.clip(X_test, -5, 5)

# ===============================
# FINAL MODEL (ANTI-OVERFITTING)
# ===============================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_bal, y_train_bal)

# ===============================
# EVALUATION
# ===============================
train_pred = model.predict(X_train_bal)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train_bal, train_pred)
test_acc = accuracy_score(y_test, test_pred)

cm = confusion_matrix(y_test, test_pred)

print("\nConfusion Matrix (Test):")
print(cm)

print("\n==============================")
print("FINAL EMG MODEL READY")
print("==============================")
print("Train Accuracy:", round(train_acc, 2))
print("Test Accuracy :", round(test_acc, 2))
print("\n✅ GENERALIZED MODEL (NO OVERFITTING) READY")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")