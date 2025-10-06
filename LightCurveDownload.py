from pathlib import Path
import pandas as pd
from pyvo.dal import TAPService
from pyasassn.client import SkyPatrolClient

OUT = Path("asassn_lcs"); OUT.mkdir(exist_ok=True)

MIN_PROB = 0.99
PER_CLASS = 5000
TABLE = 'II/366/catv2021'  # VizieR ASAS-SN catalog
TAP = TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")

FAMS = {
    "CEP":  ["CEP","DCEP","DCEPS","ACEP","CW","CWA","CWB","BLBOO"],
    "DSCT": ["DSCT","DSCUT","HADS"],
    "EB":   ["ECL","EA","EB","EW"],
    "M":    ["M","MIRA"],
    "RR":   ["RR","RRAB","RRC","RRD","RRE"],
    "SR":   ["SR","SRA","SRB","SRC","SRD"],
}
PARENTS = ["CEP","DSCT","EB","M","RR","SR"]


def fetch_gaia_ids(parent: str) -> list[int]:
    """Pull Gaia DR3 source_id (VizieR column GaiaDR3) for one parent class with Prob >= MIN_PROB."""
    like = " OR ".join([f'"Type" LIKE \'{p}%\'' for p in FAMS[parent]])
    q = f'''
      SELECT "GaiaDR3","ASASSN-V","Type","Prob"
      FROM "{TABLE}"
      WHERE "Prob" >= {MIN_PROB} AND ({like}) AND "GaiaDR3" IS NOT NULL
    '''
    t = TAP.search(q).to_table()
    if len(t) == 0:
        return []
    df = t.to_pandas()
    # keep unique, drop NaNs
    ids = pd.to_numeric(df["GaiaDR3"], errors="coerce").dropna().astype("int64").unique().tolist()
    # simple random sample up to PER_CLASS
    if len(ids) > PER_CLASS:
        ids = pd.Series(ids).sample(PER_CLASS, random_state=42).tolist()
    return ids


def chunks(lst, n=400):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def gaia_to_asassn(gaia_ids: list[int]) -> list[int]:
    """Use SkyPatrol 'stellar_main' to map Gaia DR3 IDs -> numeric asas_sn_id."""
    client = SkyPatrolClient()
    asas_ids = []
    for group in chunks(gaia_ids, 1000):
        res = client.query_list(group, catalog='stellar_main', id_col='gaia_id')
        # result is a CSV saved to disk OR a small table; handle both
        if isinstance(res, list):  # files on disk
            for fp in res:
                df = pd.read_csv(fp)
                if "asas_sn_id" in df.columns:
                    asas_ids.extend(df["asas_sn_id"].dropna().astype("int64").tolist())
        else:
            # some versions return a pandas-like object
            df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
            if "asas_sn_id" in df.columns:
                asas_ids.extend(df["asas_sn_id"].dropna().astype("int64").tolist())
    # unique
    return sorted(set(asas_ids))


def download_by_asas_id(asas_ids: list[int]) -> int:
    """Download light curves from aavsovsx by numeric asas_sn_id."""
    client = SkyPatrolClient()
    total = 0
    for group in chunks(asas_ids, 300):
        in_list = ",".join(str(x) for x in group)
        q = f"""
          SELECT asas_sn_id, variability_type, period
          FROM aavsovsx
          WHERE asas_sn_id IN ({in_list})
            AND period IS NOT NULL
        """
        files = client.adql_query(
            query_str=q,
            download=True,
            save_dir=str(OUT),
            file_format="csv",
            threads=8
        )
        total += len(files or [])
    return total


if __name__ == "__main__":
    grand = 0
    for parent in PARENTS:
        gaia_ids = fetch_gaia_ids(parent)
        print(f"{parent}: {len(gaia_ids)} Gaia IDs (Prob ≥ {MIN_PROB})")
        if not gaia_ids:
            continue
        asas_ids = gaia_to_asassn(gaia_ids)
        print(f"{parent}: {len(asas_ids)} matched asas_sn_id")
        got = download_by_asas_id(asas_ids)
        print(f"{parent}: downloaded {got} files")
        grand += got
    print(f"Done. Total downloaded: {grand}. Saved in {OUT.resolve()}")
