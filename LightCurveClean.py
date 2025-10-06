import re
import numpy as np
import pandas as pd
from pathlib import Path

# subclass → parent mapping to standardize labels
PARENTS = {
    "RR":   ["RR", "RRAB", "RRC", "RRD", "RRE"],
    "CEP":  ["CEP", "DCEP", "DCEPS", "ACEP", "CW", "CWA", "CWB", "BLBOO"],
    "DSCT": ["DSCT", "HADS"],
    "EB":   ["EB", "EA", "EW", "ECL"],
    "SR":   ["SR", "SRA", "SRB", "SRC", "SRD"],
    "M":    ["M"],
}


def to_parent_class(vtype: str | None) -> str | None:
    if not vtype:
        return None
    tok = re.sub(r"[^A-Z0-9]", "", str(vtype).upper())
    for parent, subs in PARENTS.items():
        for s in subs:
            if tok.startswith(s):
                return parent
    return tok


def _grab(metadata: str, field: str):
    m = re.search(rf'["\']?{field}["\']?\s*:\s*("?)([^,"\s]+)\1', metadata)
    return m.group(2) if m else None


def _drop_3sigma_outliers(df: pd.DataFrame, mag_col: str) -> pd.DataFrame:
    m = df[mag_col].to_numpy(np.float64)
    mu, sd = np.nanmean(m), np.nanstd(m)
    if not np.isfinite(sd) or sd == 0:
        return df
    zmask = np.abs(m - mu) <= 3.0 * sd
    return df[zmask]


def clean_asassn_csv(
    in_path: str,
    out_dir: str | None = None,
):
    """
    Read one original ASAS-SN CSV from asassn_lcs/,
    filter rows (g-band, drop mag_err sentinels, 3σ outliers),
    require >= min_points, and write to asassn_lcs_clean/ with metadata preserved.
    """
    in_p = Path(in_path)
    if out_dir is None:
        out_dir = in_p.parent.with_name("asassn_lcs_clean")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_p.stem}_clean.csv"

    # --- metadata (kept as first line) ---
    with open(in_p, "r", encoding="utf-8") as f:
        metadata = f.readline().strip()
    period_raw = _grab(metadata, "period")
    period = float(period_raw) if period_raw is not None else None
    vtype_parent = to_parent_class(_grab(metadata, "variability_type"))

    # --- table (headers on line 2) ---
    df = pd.read_csv(in_p, skiprows=1)
    df.columns = [c.strip().lower() for c in df.columns]

    # coerce numerics
    for c in ["jd", "mag", "mag_err", "flux", "flux_err", "limit", "fwhm", "period"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute phase if needed
    if "phase" not in df.columns and period is not None and "jd" in df.columns:
        df["phase"] = (df["jd"] / period) % 1.0

    # g-band only
    if "phot_filter" in df.columns:
        df = df[df["phot_filter"].astype(str).str.lower() == "g"]

    # drop rows with sentinel/invalid mag_err (e.g., 9.9 / 99.9)
    if "mag_err" in df.columns:
        df = df[df["mag_err"].notna() & (df["mag_err"] < 9.0)]

    # choose magnitude column (no alignment)
    mag_col = "mag"
    df = df[df[mag_col].notna()]

    # optional 3σ single-point outlier removal
    if not df.empty:
        df = _drop_3sigma_outliers(df, mag_col)

    # keep only essential columns
    if vtype_parent is not None:
        df["variability_type"] = vtype_parent
    if period is not None:
        df["period"] = period

    cols = [c for c in [
        "jd","phase","mag","mag_err","flux","flux_err","limit",
        "fwhm","phot_filter","quality","variability_type","period"
    ] if c in df.columns]
    out = df[cols].copy()

    # require at least min_points
    if len(out) < 50:
        return None

    # write (metadata + CSV)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(metadata + "\n")
        out.to_csv(f, index=False)

    return str(out_path)


def clean_asassn_dir(in_dir: str = "asassn_lcs", out_dir: str = "asassn_lcs_clean", **kwargs):
    in_d, out_d = Path(in_dir), Path(out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    outs = []
    for csv in in_d.glob("*.csv"):
        p = clean_asassn_csv(str(csv), out_dir=str(out_d))
        if p is not None:
            outs.append(p)
    return outs


# from clean_asassn import clean_asassn_dir
clean_asassn_dir("asassn_lcs", "asassn_lcs_clean")
