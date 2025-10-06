from pathlib import Path
from pyasassn.client import SkyPatrolClient

OUT = Path("asassn_lcs_unlabeled"); OUT.mkdir(parents=True, exist_ok=True)

QUERY = """
  SELECT asas_sn_id, variability_type, period
  FROM aavsovsx
  WHERE period IS NOT NULL
  AND variability_type IS NULL
  LIMIT 5
"""

if __name__ == "__main__":
    client = SkyPatrolClient()
    files = client.adql_query(
        query_str=QUERY,
        download=True,
        save_dir=str(OUT),
        file_format="csv",
        threads=8
    )
    print(f"Downloaded {len(files or [])} light curves into {OUT.resolve()}")
