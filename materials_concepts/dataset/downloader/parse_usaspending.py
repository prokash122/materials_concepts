import zipfile
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Only keep these columns from the 297-column CSV
KEEP_COLUMNS = [
    "contract_award_unique_key",
    "transaction_description",
    "prime_award_base_transaction_description",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "federal_action_obligation",
    "total_dollars_obligated",
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "funding_agency_name",
    "funding_sub_agency_name",
    "recipient_name",
    "recipient_state_name",
    "naics_code",
    "naics_description",
    "product_or_service_code_description",
    "research",
    "program_acronym",
    "major_program",
    "award_type",
]

# Materials science NAICS code prefixes
MATERIALS_NAICS_PREFIXES = [
    "3311", "3312", "3313", "3314", "3315",  # Primary Metal Manufacturing
    "3321", "3322", "3323", "3324", "3325", "3326", "3327", "3328", "3329",  # Fabricated Metal
    "3271", "3272", "3273", "3274", "3279",  # Nonmetallic Mineral (ceramics, glass)
    "3251", "3252", "3253", "3254", "3255", "3256", "3259",  # Chemical Manufacturing
    "3261", "3262",  # Plastics and Rubber
    "3341", "3342", "3343", "3344", "3345", "3346",  # Computer/Electronic (semiconductors)
    "5417",  # R&D in Physical/Engineering Sciences
]

# Sub-agencies of interest for materials/manufacturing research
MATERIALS_AGENCIES = [
    "air force research laboratory",
    "army research laboratory",
    "army research office",
    "office of naval research",
    "defense advanced research projects agency",
    "department of energy",
    "national institute of standards",
    "national science foundation",
    "army materiel command",
    "naval research laboratory",
    "air force office of scientific research",
]


def is_materials_contract(row: pd.Series, lookup_concepts: set | None) -> bool:
    naics = str(row.get("naics_code", "") or "")
    sub_agency = str(row.get("awarding_sub_agency_name", "") or "").lower()
    naics_desc = str(row.get("naics_description", "") or "").lower()
    description = str(row.get("transaction_description", "") or "").lower()

    # filter by NAICS code (most reliable)
    for prefix in MATERIALS_NAICS_PREFIXES:
        if naics.startswith(prefix):
            return True

    # filter by sub-agency name
    for agency in MATERIALS_AGENCIES:
        if agency in sub_agency:
            return True

    # filter by lookup concepts in description
    if lookup_concepts:
        return any(c in description or c in naics_desc for c in lookup_concepts)

    return False


def parse_csv_from_zip(zip_path: Path, lookup_concepts: set | None, chunksize: int = 10000) -> pd.DataFrame:
    results = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        logger.info(f"  Found {len(csv_files)} CSV file(s) in {zip_path.name}")

        for csv_file in csv_files:
            with zf.open(csv_file) as f:
                try:
                    for chunk in tqdm(
                        pd.read_csv(
                            f,
                            usecols=lambda c: c in KEEP_COLUMNS,
                            chunksize=chunksize,
                            low_memory=False,
                            encoding_errors="replace",
                        ),
                        desc=f"  {csv_file[:40]}",
                        unit=f" chunks({chunksize})",
                    ):
                        filtered = chunk[chunk.apply(
                            lambda row: is_materials_contract(row, lookup_concepts), axis=1
                        )]
                        results.append(filtered)
                except Exception as e:
                    logger.warning(f"  Error reading {csv_file}: {e}")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    # use base description if transaction description is empty
    df["abstract"] = df["transaction_description"].fillna("")
    mask = df["abstract"].str.strip() == ""
    df.loc[mask, "abstract"] = df.loc[mask, "prime_award_base_transaction_description"].fillna("")

    return pd.DataFrame({
        "id":               df["contract_award_unique_key"],
        "display_name":     df["abstract"].str[:200],
        "abstract":         df["abstract"],
        "publication_date": pd.to_datetime(df["period_of_performance_start_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "exp_date":         df["period_of_performance_current_end_date"],
        "amount":           pd.to_numeric(df["federal_action_obligation"], errors="coerce").fillna(0),
        "total_amount":     pd.to_numeric(df["total_dollars_obligated"], errors="coerce").fillna(0),
        "agency":           df["awarding_agency_name"],
        "division":         df["awarding_sub_agency_name"],
        "institution":      df["recipient_name"],
        "state":            df["recipient_state_name"],
        "naics":            df["naics_description"],
        "program":          df["program_acronym"].fillna("") + " " + df["major_program"].fillna(""),
        "research_type":    df["research"],
        "is_retracted":     False,
        "is_paratext":      False,
    })


@click.command("Parse USASpending.gov bulk contract ZIP files for materials science & manufacturing.")
@click.argument("inputs", nargs=-1, required=True)
@click.option(
    "--out",
    default="data/table/usaspending-materials.works.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--lookup-path",
    default="data/table/lookup/lookup.M.csv",
    show_default=True,
    help="Path to lookup.M.csv for additional keyword filtering.",
)
@click.option(
    "--min-concept-count",
    default=10,
    show_default=True,
    type=int,
    help="Only use lookup concepts with count >= this value.",
)
@click.option(
    "--min-abstract-length",
    default=50,
    show_default=True,
    type=int,
    help="Minimum abstract length to keep a record.",
)
def parse_usaspending(
    inputs: tuple,
    out: str,
    lookup_path: str,
    min_concept_count: int,
    min_abstract_length: int,
):
    Path(out).parent.mkdir(parents=True, exist_ok=True)

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
        logger.info(f"Loaded {len(lookup_concepts)} concepts for filtering")

    all_dfs = []
    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        logger.info(f"Processing: {path.name}")
        df = parse_csv_from_zip(path, lookup_concepts)
        if df.empty:
            logger.warning(f"  No matching records in {path.name}")
            continue

        logger.info(f"  {len(df)} materials records found in {path.name}")
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No records found. Check your input files.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = to_pipeline_format(merged)
    merged = merged[
        merged["abstract"].notna() &
        (merged["abstract"].str.len() >= min_abstract_length)
    ]
    merged = merged.drop_duplicates(subset="id").reset_index(drop=True)
    merged.to_csv(out, index=False)

    logger.info(f"Saved {len(merged)} unique contracts to '{out}'")
    logger.info(f"By agency:\n{merged['agency'].value_counts().head(10).to_string()}")
    logger.info(f"By sub-agency:\n{merged['division'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    parse_usaspending()
