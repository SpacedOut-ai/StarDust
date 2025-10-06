import re
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras

UNLABELED_DIR = Path("asassn_lcs_unlabeled")
MODEL_PATH = "lightcurve_cnn.keras"
LABEL_MAP_TXT = "label_map.txt"   # optional
N_POINTS = 512

# Fallback class order used in our training
FALLBACK_CLASSES = ["CEP", "DSCT", "EB", "M", "RR", "SR"]


def load_labels():
    p = Path(LABEL_MAP_TXT)
    if not p.exists():
        return FALLBACK_CLASSES
    classes = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        # lines look like: "0\tCEP"
        parts = line.split()
        classes.append(parts[-1])
    return classes


def read_period_from_metadata(csv_path: Path) -> float:
    with open(csv_path, "r", encoding="utf-8") as f:
        meta = f.readline().strip()
    m = re.search(r'["\']?period["\']?\s*:\s*("?)([0-9]*\.?[0-9]+)\1', meta, flags=re.I)
    if not m:
        raise ValueError(f"No period found in metadata: {csv_path.name}")
    return float(m.group(2))


def periodic_interp(phase: np.ndarray, mag: np.ndarray, n_points=N_POINTS) -> np.ndarray:
    phase = np.mod(phase, 1.0).astype(np.float32)
    order = np.argsort(phase)
    p, y = phase[order], mag[order]
    if p.size < 2:
        return np.full(n_points, np.nan, dtype=np.float32)
    # wrap for periodic interpolation
    p_ext = np.concatenate([p - 1.0, p, p + 1.0])
    y_ext = np.concatenate([y, y, y])
    grid = np.linspace(0.0, 1.0, n_points, endpoint=False).astype(np.float32)
    return np.interp(grid, p_ext, y_ext).astype(np.float32)


def preprocess_one(csv_path: Path) -> np.ndarray | None:
    """Return a (1, 512, 1) array ready for the CNN, or None if unusable."""
    period = read_period_from_metadata(csv_path)
    # data table begins on line 2
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip().lower() for c in df.columns]

    # minimal cleaning: drop sentinel mag_err >= 9 (if present), drop NaNs
    if "mag_err" in df.columns:
        df.loc[df["mag_err"] >= 9.0, "mag_err"] = np.nan

    # If phot_filter exists, keep g-band only (optional but helps)
    if "phot_filter" in df.columns:
        df = df[df["phot_filter"].astype(str).str.lower() == "g"]

    # compute phase
    phase = (df["jd"] / period) % 1.0
    mag   = df["mag"]

    mask = np.isfinite(phase) & np.isfinite(mag)
    if mask.sum() < 2:
        return None

    y = periodic_interp(phase[mask].to_numpy(np.float32),
                        mag[mask].to_numpy(np.float32),
                        N_POINTS)

    if not np.isfinite(y).any():
        return None

    # per-curve z-score (same as training)
    mu, sd = y.mean(), y.std() + 1e-6
    y = (y - mu) / sd

    return y.reshape(1, N_POINTS, 1).astype(np.float32)


def topk(probs: np.ndarray, k: int = 3):
    idx = np.argsort(-probs)[:k]
    return idx, probs[idx]


if __name__ == "__main__":
    classes = load_labels()
    model = keras.models.load_model(MODEL_PATH)

    rows = []
    for fp in sorted(UNLABELED_DIR.glob("*.csv")):
        x = preprocess_one(fp)
        if x is None:
            rows.append({"file": fp.name, "pred": "SKIPPED", "p": "", "top3": ""})
            continue

        p = model.predict(x, verbose=0)[0]   # softmax
        pred_idx = int(np.argmax(p))
        pred_cls = classes[pred_idx]
        pred_p   = float(p[pred_idx])

        top_idx, top_p = topk(p, k=3)
        top3 = "; ".join([f"{classes[i]}:{top_p[j]:.3f}" for j, i in enumerate(top_idx)])

        rows.append({"file": fp.name, "pred": pred_cls, "p": f"{pred_p:.3f}", "top3": top3})

    out_df = pd.DataFrame(rows)
    out_df.to_csv("unlabeled_predictions.csv", index=False)
    print(out_df.head(20))
    print("\nSaved predictions -> unlabeled_predictions.csv")
