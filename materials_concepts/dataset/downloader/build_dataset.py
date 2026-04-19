"""
Build train/test/predict CSVs directly from USASpending Assistance ZIP files.

Year is extracted from the filename (FY2017_... → 2017).
Each ZIP is routed to train, test, or predict based on --train-end / --test-end.
"""
import re
import zipfile
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Key columns to extract from 112-column CSV
KEEP_COLUMNS = [
    "assistance_award_unique_key",
    "transaction_description",
    "prime_award_base_transaction_description",
    "funding_opportunity_goals_text",
    "cfda_number",
    "cfda_title",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "federal_action_obligation",
    "total_obligated_amount",
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "recipient_name",
    "recipient_state_name",
    "assistance_type_description",
    "funding_opportunity_number",
]

MATERIALS_CFDA_PREFIXES = [
    "47.041",  # NSF Engineering
    "47.049",  # NSF Mathematical & Physical Sciences (DMR)
    "47.076",  # NSF STEM Education
    "81.049",  # DOE Basic Energy Sciences
    "81.057",  # DOE Advanced Manufacturing
    "81.086",  # DOE Conservation R&D
    "81.089",  # DOE Fossil Energy R&D
    "81.121",  # DOE Nuclear Energy R&D
    "12.800",  # DOD Defense Research Sciences
    "12.630",  # DOD Basic Research
    "12.300",  # DOD Research
    "11.609",  # NIST Manufacturing Extension Partnership
    "11.616",  # NIST Advanced Manufacturing Technology
    "43.001",  # NASA Science
    "43.002",  # NASA Aeronautics
]

MATERIALS_CFDA_KEYWORDS = [
    "materials", "manufacturing", "engineering", "physics",
    "chemistry", "metallurgy", "ceramic", "polymer",
    "nanotechnology", "semiconductor", "energy science",
    "basic research", "basic energy", "advanced manufacturing",
]

MATERIALS_SUBAGENCIES = [
    "air force research laboratory",
    "army research laboratory",
    "army research office",
    "office of naval research",
    "defense advanced research projects agency",
    "naval research laboratory",
    "air force office of scientific research",
    "national science foundation",
    "office of science",
    "national institute of standards",
    "advanced research projects agency",
]


def extract_year(filename: str) -> int | None:
    """Extract fiscal year from filename like FY2017_All_Assistance_..."""
    match = re.search(r"FY(\d{4})", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def is_materials_grant(row: pd.Series, lookup_concepts: set | None) -> bool:
    cfda_num = str(row.get("cfda_number", "") or "")
    cfda_title = str(row.get("cfda_title", "") or "").lower()
    sub_agency = str(row.get("awarding_sub_agency_name", "") or "").lower()
    description = str(row.get("transaction_description", "") or "").lower()

    for prefix in MATERIALS_CFDA_PREFIXES:
        if cfda_num.startswith(prefix):
            return True
    for kw in MATERIALS_CFDA_KEYWORDS:
        if kw in cfda_title:
            return True
    for agency in MATERIALS_SUBAGENCIES:
        if agency in sub_agency:
            return True
    if lookup_concepts and description:
        return any(c in description for c in lookup_concepts)
    return False


def parse_zip(zip_path: Path, lookup_concepts: set | None, chunksize: int = 10000) -> pd.DataFrame:
    results = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        for csv_file in csv_files:
            with zf.open(csv_file) as f:
                try:
                    chunks = pd.read_csv(
                        f,
                        usecols=lambda c: c in KEEP_COLUMNS,
                        chunksize=chunksize,
                        low_memory=False,
                        encoding_errors="replace",
                    )
                    for chunk in tqdm(chunks, desc=f"  {csv_file[-40:]}", unit=" chunks"):
                        filtered = chunk[chunk.apply(
                            lambda row: is_materials_grant(row, lookup_concepts), axis=1
                        )]
                        if not filtered.empty:
                            results.append(filtered)
                except Exception as e:
                    logger.warning(f"  Error reading {csv_file}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    desc1 = df["transaction_description"].fillna("")
    desc2 = df["prime_award_base_transaction_description"].fillna("")
    goals = df["funding_opportunity_goals_text"].fillna("")

    abstract = desc1.where(desc1.str.len() >= desc2.str.len(), desc2)
    has_goals = goals.str.len() > 10
    abstract[has_goals] = abstract[has_goals] + " " + goals[has_goals]

    return pd.DataFrame({
        "id":               df["assistance_award_unique_key"],
        "abstract":         abstract,
        "publication_date": pd.to_datetime(
            df["period_of_performance_start_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "exp_date":         df["period_of_performance_current_end_date"],
        "amount":           pd.to_numeric(df["federal_action_obligation"], errors="coerce").fillna(0),
        "total_amount":     pd.to_numeric(df["total_obligated_amount"], errors="coerce").fillna(0),
        "agency":           df["awarding_agency_name"],
        "division":         df["awarding_sub_agency_name"],
        "program":          df["cfda_title"],
        "cfda":             df["cfda_number"],
        "opportunity_number": df["funding_opportunity_number"],
        "institution":      df["recipient_name"],
        "state":            df["recipient_state_name"],
        "grant_type":       df["assistance_type_description"],
        "is_retracted":     False,
        "is_paratext":      False,
    })


@click.command("Build train/test/predict datasets directly from USASpending Assistance ZIPs.")
@click.argument("inputs", nargs=-1, required=True)
@click.option("--train-end", default=2020, show_default=True, type=int,
              help="Last year (inclusive) for training set.")
@click.option("--test-end", default=2024, show_default=True, type=int,
              help="Last year (inclusive) for test set.")
@click.option("--out-dir", default="data/table", show_default=True,
              help="Output directory.")
@click.option("--lookup-path", default="data/table/lookup/lookup.M.csv", show_default=True)
@click.option("--min-concept-count", default=10, show_default=True, type=int)
@click.option("--min-abstract-length", default=100, show_default=True, type=int)
def build_dataset(
    inputs: tuple,
    train_end: int,
    test_end: int,
    out_dir: str,
    lookup_path: str,
    min_concept_count: int,
    min_abstract_length: int,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # load lookup concepts
    lookup_concepts = None
    if lookup_path and Path(lookup_path).exists():
        ldf = pd.read_csv(lookup_path)
        if "count" in ldf.columns:
            ldf = ldf[ldf["count"] >= min_concept_count]
        lookup_concepts = set(
            c.lower() for c in ldf["concept"].dropna().tolist()
            if len(c.strip()) >= 8
        )
        logger.info(f"Loaded {len(lookup_concepts)} lookup concepts")

    train_dfs, test_dfs, pred_dfs = [], [], []

    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        year = extract_year(path.name)
        if year is None:
            logger.warning(f"Could not extract year from '{path.name}' — skipping")
            continue

        if year <= train_end:
            split = "TRAIN"
        elif year <= test_end:
            split = "TEST"
        else:
            split = "PREDICT"

        logger.info(f"[{split}] Processing FY{year}: {path.name}")
        df = parse_zip(path, lookup_concepts)
        if df.empty:
            logger.warning(f"  No matching records in {path.name}")
            continue

        df = to_pipeline_format(df)
        df = df[df["abstract"].notna() & (df["abstract"].str.len() >= min_abstract_length)]
        logger.info(f"  {len(df):,} records after filtering")

        if split == "TRAIN":
            train_dfs.append(df)
        elif split == "TEST":
            test_dfs.append(df)
        else:
            pred_dfs.append(df)

    def save(dfs: list, name: str):
        if not dfs:
            logger.warning(f"No records for {name}")
            return
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="id").reset_index(drop=True)
        out_file = out_path / f"assistance-{name}.works.csv"
        merged.to_csv(out_file, index=False)
        logger.info(f"Saved {len(merged):,} unique records → {out_file}")
        logger.info(f"  By agency:\n{merged['agency'].value_counts().head(5).to_string()}")

    save(train_dfs, "train")
    save(test_dfs, "test")
    save(pred_dfs, "predict")


if __name__ == "__main__":
    build_dataset()
