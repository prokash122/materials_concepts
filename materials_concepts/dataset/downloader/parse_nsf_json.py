import json
import zipfile
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

# NSF division/directorate filters for materials science & manufacturing
MATERIALS_DIVISIONS = [
    "Division of Materials Research",
    "Civil, Mechanical and Manufacturing Innovation",
    "Division of Civil, Mechanical and Manufacturing Innovation",
    "Advanced Manufacturing",
    "Materials Engineering and Processing",
    "Metals, Minerals, and Mining",
    "Ceramics",
    "Polymers",
    "Solid State and Materials Chemistry",
    "Condensed Matter Physics",
    "Biomaterials",
    "Electronic and Photonic Materials",
    "Division of Materials Research",
    "Division of Manufacturing Innovation",
]

# Only the most specific materials/manufacturing NSF divisions
MATERIALS_DIV_ABBR = ["DMR", "CMMI"]

MATERIALS_KEYWORDS = [
    "material", "manufacturing", "ceramic", "polymer", "composite",
    "alloy", "coating", "thin film", "nanostructure", "semiconductor",
    "piezoelectric", "ferroelectric", "crystal", "microstructure",
]


def get_pi_name(pi_list: list) -> str:
    if not pi_list:
        return ""
    for pi in pi_list:
        if pi.get("pi_role", "").lower() == "principal investigator":
            return pi.get("pi_full_name", "")
    return pi_list[0].get("pi_full_name", "")


def get_program_names(pgm_ele: list) -> str:
    if not pgm_ele:
        return ""
    return "; ".join(p.get("pgm_ele_name", "") for p in pgm_ele)


def is_materials_related(record: dict, filter_by_division: bool, lookup_concepts: set | None = None) -> bool:
    if not filter_by_division:
        return True

    abstract = (record.get("awd_abstract_narration") or "").lower()
    title = (record.get("awd_titl_txt") or "").lower()
    text = abstract + " " + title

    # primary filter: lookup.M.csv concepts must appear in abstract or title
    if lookup_concepts:
        return any(c in text for c in lookup_concepts)

    # fallback if no lookup provided: use division + hardcoded keywords
    div_name = (record.get("org_div_long_name") or "").lower()
    div_abbr = (record.get("div_abbr") or "").upper()

    for div in MATERIALS_DIVISIONS:
        if div.lower() in div_name:
            return True
    if div_abbr in MATERIALS_DIV_ABBR:
        return True
    return any(kw in text for kw in MATERIALS_KEYWORDS)


def parse_record(record: dict) -> dict:
    pi_list = record.get("pi", [])
    inst = record.get("inst", {})

    return {
        "id":               record.get("awd_id", ""),
        "display_name":     record.get("awd_titl_txt", ""),
        "abstract":         record.get("awd_abstract_narration") or "",
        "publication_date": record.get("awd_eff_date", ""),
        "exp_date":         record.get("awd_exp_date", ""),
        "amount":           record.get("awd_amount", 0),
        "total_amount":     record.get("tot_intn_awd_amt", 0),
        "division":         record.get("org_div_long_name", ""),
        "div_abbr":         record.get("div_abbr", ""),
        "directorate":      record.get("org_dir_long_name", ""),
        "program":          get_program_names(record.get("pgm_ele", [])),
        "pi":               get_pi_name(pi_list),
        "institution":      inst.get("inst_name", ""),
        "state":            inst.get("inst_state_name", ""),
        "is_retracted":     False,
        "is_paratext":      False,
    }


def load_json_from_zip(zip_path: Path) -> list[dict]:
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        for fname in tqdm(json_files, desc=f"Parsing {zip_path.name}", unit=" files"):
            with zf.open(fname) as f:
                try:
                    record = json.load(f)
                    records.append(record)
                except Exception as e:
                    logger.debug(f"Skipping {fname}: {e}")
    return records


def load_json_from_file(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    # handle both single record and list of records
    return data if isinstance(data, list) else [data]


@click.command("Parse NSF bulk JSON files into pipeline-compatible CSV.")
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
)
@click.option(
    "--out",
    default="data/table/nsf-materials-manufacturing.works.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--lookup-path",
    default=None,
    help="Optional: path to lookup.M.csv to filter abstracts by known concepts.",
)
@click.option(
    "--min-concept-length",
    default=8,
    show_default=True,
    type=int,
    help="Minimum character length of lookup concepts used for filtering (avoids short ambiguous matches like 'CO2', 'OH').",
)
@click.option(
    "--min-concept-count",
    default=5,
    show_default=True,
    type=int,
    help="Only use lookup concepts with count >= this value.",
)
@click.option(
    "--filter-by-division",
    default=True,
    show_default=True,
    type=bool,
    help="Filter awards to materials science & manufacturing divisions only.",
)
@click.option(
    "--min-abstract-length",
    default=100,
    show_default=True,
    type=int,
    help="Minimum abstract length in characters.",
)
def parse_nsf_json(
    inputs: tuple,
    out: str,
    lookup_path: str | None,
    min_concept_length: int,
    min_concept_count: int,
    filter_by_division: bool,
    min_abstract_length: int,
):
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # load lookup concepts if provided — filter short/rare concepts
    lookup_concepts = None
    if lookup_path:
        lookup_df = pd.read_csv(lookup_path)
        if "count" in lookup_df.columns:
            lookup_df = lookup_df[lookup_df["count"] >= min_concept_count]
        lookup_concepts = set(
            c.lower() for c in lookup_df["concept"].dropna().tolist()
            if len(c.strip()) >= min_concept_length
        )
        logger.info(f"Loaded {len(lookup_concepts)} concepts from lookup (min_length={min_concept_length}, min_count={min_concept_count})")

    all_records = []

    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        logger.info(f"Loading: {path.name}")
        if path.suffix == ".zip":
            raw = load_json_from_zip(path)
        elif path.suffix == ".json":
            raw = load_json_from_file(path)
        else:
            logger.warning(f"Unsupported file type: {path.suffix} — skipping")
            continue

        logger.info(f"  Loaded {len(raw)} raw records from {path.name}")
        all_records.extend(raw)

    if not all_records:
        logger.error("No records loaded. Check your input files.")
        return

    logger.info(f"Total raw records: {len(all_records)}")

    # parse and filter
    parsed = []
    for record in tqdm(all_records, desc="Parsing records"):
        if not is_materials_related(record, filter_by_division, lookup_concepts):
            continue
        parsed.append(parse_record(record))

    logger.info(f"After division filter: {len(parsed)} records")

    df = pd.DataFrame(parsed)

    # normalize date
    df["publication_date"] = pd.to_datetime(
        df["publication_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # filter empty/short abstracts
    df = df[df["abstract"].notna() & (df["abstract"].str.len() >= min_abstract_length)]
    logger.info(f"After abstract filter: {len(df)} records")


    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df.to_csv(out, index=False)

    logger.info(f"Saved {len(df)} awards to '{out}'")
    logger.info(f"Divisions found: {df['division'].value_counts().head(10).to_dict()}")


if __name__ == "__main__":
    parse_nsf_json()
