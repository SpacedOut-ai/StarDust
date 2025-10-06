import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import tensorflow as tf

DATA_DIR = Path("asassn_lcs_clean")
CLASSES = ["CEP", "DSCT", "EB", "M", "RR", "SR"]
N_POINTS = 512
BATCH = 16
EPOCHS = 100
LR = 1e-4
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def read_clean_csv(p: Path) -> pd.DataFrame:
    with open(p, "r", encoding="utf-8") as f: _ = f.readline()
    df = pd.read_csv(p, skiprows=1)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def ensure_phase(df: pd.DataFrame) -> np.ndarray:
    if "phase" in df.columns:
        return df["phase"].to_numpy(np.float32)
    per = float(df["period"].dropna().iloc[0])
    return ((df["jd"] / per) % 1.0).to_numpy(np.float32)


def periodic_interp(phase: np.ndarray, mag: np.ndarray, n_points=N_POINTS):
    phase = np.mod(phase, 1.0).astype(np.float32)
    order = np.argsort(phase)
    p, y = phase[order], mag[order]
    if p.size < 2:
        return np.full(n_points, np.nan, dtype=np.float32)
    p_ext = np.concatenate([p - 1.0, p, p + 1.0])
    y_ext = np.concatenate([y, y, y])
    grid = np.linspace(0.0, 1.0, n_points, endpoint=False).astype(np.float32)
    return np.interp(grid, p_ext, y_ext).astype(np.float32)


def to_series_512(df: pd.DataFrame):
    # ALWAYS use raw magnitude (no alignment)
    mag = df["mag"].to_numpy(np.float32)
    phase = ensure_phase(df)
    mask = np.isfinite(phase) & np.isfinite(mag)
    phase, mag = phase[mask], mag[mask]
    y = periodic_interp(phase, mag, N_POINTS)
    mu, sd = y.mean(), y.std() + 1e-6
    y = (y - mu) / sd
    return y.astype(np.float32)  # [512]


def build_cnn(n_classes: int, length: int = N_POINTS):
    inp = keras.Input(shape=(length, 1))

    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)

    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# -------- load data -------
files = sorted(glob.glob(str(DATA_DIR / "*_clean.csv")))
X_list, y_list = [], []
for f in files:
    df = read_clean_csv(Path(f))
    if df.empty or "variability_type" not in df.columns:
        continue
    label = str(df["variability_type"].iloc[0]).upper()
    if label not in CLASSES:
        continue
    series = to_series_512(df)
    if not np.isfinite(series).any():
        continue
    X_list.append(series)
    y_list.append(CLASSES.index(label))

X = np.stack(X_list, axis=0)[..., None]  # [N, 512, 1]
y = np.array(y_list, dtype=np.int32)
print(f"Loaded {len(X)} curves from {DATA_DIR}")

# stratified train/val/test
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.1875, random_state=SEED, stratify=y_tmp
)  # final ≈ 65/15/20

# class weights
cw = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASSES)), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(cw)}

# train
model = build_cnn(n_classes=len(CLASSES), length=N_POINTS)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    verbose=1,
)

# evaluate
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3, zero_division=0))

# save
model.save("lightcurve_cnn.keras")
with open("label_map.txt", "w", encoding="utf-8") as f:
    for i, c in enumerate(CLASSES):
        f.write(f"{i}\t{c}\n")
print("Saved model -> lightcurve_cnn.keras and label_map.txt")
