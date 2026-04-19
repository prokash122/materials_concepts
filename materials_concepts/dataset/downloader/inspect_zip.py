"""Quick inspector: prints columns and sample rows from a USASpending ZIP."""
import zipfile
import sys
from pathlib import Path
import pandas as pd

zip_path = Path(sys.argv[1])
nrows = int(sys.argv[2]) if len(sys.argv) > 2 else 3

with zipfile.ZipFile(zip_path, "r") as zf:
    csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
    print(f"\nFound {len(csv_files)} CSV file(s): {csv_files}\n")

    for csv_file in csv_files[:1]:  # inspect first CSV only
        with zf.open(csv_file) as f:
            df = pd.read_csv(f, nrows=nrows, low_memory=False, encoding_errors="replace")

        print(f"=== {csv_file} ===")
        print(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            sample = str(df[col].iloc[0])[:80] if not df.empty else ""
            print(f"  {col:55s} | {sample}")

        # highlight description-like columns
        desc_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["description", "abstract", "narrative", "text", "title", "summary"]
        )]
        print(f"\nDescription-like columns: {desc_cols}")
        for col in desc_cols:
            print(f"\n--- {col} ---")
            for val in df[col].dropna().head(2):
                print(f"  {str(val)[:300]}")
