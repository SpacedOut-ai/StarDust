import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_one_light_curve(csv_path: str):
    # 1) read period from metadata (line 1)
    with open(csv_path, "r", encoding="utf-8") as f:
        meta = f.readline().strip()
    period = float(re.search(r'["\']?period["\']?\s*:\s*("?)([0-9]*\.?[0-9]+)\1', meta).group(2))

    # 2) read table (headers start on line 2) and normalize column names
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip().lower() for c in df.columns]

    # 3) compute phase (one cycle) and pick magnitude
    phase = (df["jd"] / period) % 1.0
    mag = df["mag"]

    # 4) plot
    plt.figure(figsize=(7, 5))
    plt.scatter(phase, mag, s=10, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Phase (0–1)")
    plt.ylabel("Magnitude")
    plt.title(f"{Path(csv_path).stem} — Phase-folded Light Curve")
    plt.tight_layout()
    plt.show()


plot_one_light_curve(r"asassn_lcs_unlabeled\274878834682.csv")
