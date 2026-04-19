import time
from pathlib import Path

import click
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Federal agency codes on USASpending.gov
AGENCIES = {
    "DOE":   {"toptier_code": "089", "name": "Department of Energy"},
    "AFRL":  {"toptier_code": "097", "name": "Department of Defense", "subtier": "Air Force Research Laboratory"},
    "ARO":   {"toptier_code": "097", "name": "Department of Defense", "subtier": "Army Research Office"},
    "ONR":   {"toptier_code": "097", "name": "Department of Defense", "subtier": "Office of Naval Research"},
    "DARPA": {"toptier_code": "097", "name": "Department of Defense", "subtier": "Defense Advanced Research Projects Agency"},
    "NIST":  {"toptier_code": "013", "name": "Department of Commerce"},
    "NASA":  {"toptier_code": "080", "name": "National Aeronautics and Space Administration"},
    "NSF":   {"toptier_code": "422", "name": "National Science Foundation"},
}

# Materials science keywords for filtering
MATERIALS_KEYWORDS = [
    "materials science", "advanced manufacturing", "additive manufacturing",
    "piezoelectric", "ferroelectric", "smart materials", "ceramics",
    "composite materials", "thin film", "nanostructure", "semiconductor",
    "alloy", "microstructure", "corrosion", "fatigue", "fracture mechanics",
    "crystal structure", "phase transformation", "grain boundary",
    "polymer", "biomaterials", "coating", "metallurgy",
]


def build_payload(
    agency_code: str,
    keywords: list[str],
    date_start: str,
    date_end: str,
    page: int,
    limit: int = 100,
) -> dict:
    filters = {
        "award_type_codes": ["02", "03", "04", "05"],  # grants & contracts
        "time_period": [{"start_date": date_start, "end_date": date_end}],
        "agencies": [{"type": "awarding", "tier": "toptier", "toptier_code": agency_code}],
        "keywords": keywords,
    }

    return {
        "filters": filters,
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Total Outlays",
            "Description",
            "Start Date",
            "End Date",
            "Awarding Agency",
            "Awarding Sub Agency",
            "Award Type",
            "generated_internal_id",
        ],
        "page": page,
        "limit": limit,
        "sort": "Start Date",
        "order": "desc",
        "subawards": False,
    }


def fetch_page(agency_code: str, keywords: list[str], date_start: str, date_end: str, page: int) -> tuple[list, int]:
    payload = build_payload(agency_code, keywords, date_start, date_end, page)
    try:
        response = requests.post(USASPENDING_URL, json=payload, timeout=60)
        if not response.ok:
            logger.warning(f"API error {response.status_code}: {response.text[:300]}")
            return [], 0
        data = response.json()
        results = data.get("results", [])
        total = data.get("page_metadata", {}).get("total", 0)
        return results, total
    except Exception as e:
        logger.warning(f"Request failed on page {page}: {e}")
        return [], 0


def fetch_agency_awards(
    agency_key: str,
    agency_info: dict,
    keywords: list[str],
    date_start: str,
    date_end: str,
    fetch_limit: int | None,
) -> list[dict]:
    agency_code = agency_info["toptier_code"]
    all_results = []

    # batch keywords to avoid API limit (max ~20 per request)
    batch_size = 20
    keyword_batches = [keywords[i:i+batch_size] for i in range(0, len(keywords), batch_size)]
    logger.info(f"  {agency_key}: {len(keyword_batches)} keyword batches")

    for batch in keyword_batches:
        page = 1
        while True:
            results, total = fetch_page(agency_code, batch, date_start, date_end, page)
            if not results:
                break
            all_results.extend(results)
            if fetch_limit and len(all_results) >= fetch_limit:
                break
            if page * 100 >= total:
                break
            page += 1
            time.sleep(0.5)

        if fetch_limit and len(all_results) >= fetch_limit:
            break

    # deduplicate by Award ID
    seen = set()
    unique = []
    for r in all_results:
        rid = r.get("generated_internal_id") or r.get("Award ID")
        if rid not in seen:
            seen.add(rid)
            unique.append(r)

    logger.info(f"  {agency_key}: {len(unique)} unique awards found")
    return unique[:fetch_limit] if fetch_limit else unique


def to_pipeline_format(records: list[dict], agency_key: str) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "id":               r.get("generated_internal_id", ""),
            "display_name":     r.get("Description", "")[:200] if r.get("Description") else "",
            "abstract":         r.get("Description", "") or "",
            "publication_date": r.get("Start Date", ""),
            "exp_date":         r.get("End Date", ""),
            "amount":           r.get("Award Amount", 0),
            "total_amount":     r.get("Total Outlays", 0),
            "division":         r.get("Awarding Sub Agency", "") or r.get("Awarding Agency", ""),
            "institution":      r.get("Recipient Name", ""),
            "agency":           agency_key,
            "is_retracted":     False,
            "is_paratext":      False,
        })
    df = pd.DataFrame(rows)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


@click.command("Download federal research awards from USASpending.gov for multiple agencies.")
@click.option(
    "--agencies",
    default="DOE,AFRL,ARO,ONR,DARPA,NIST",
    show_default=True,
    help=f"Comma-separated agency codes. Available: {', '.join(AGENCIES.keys())}",
)
@click.option(
    "--keywords",
    default=",".join(MATERIALS_KEYWORDS),
    show_default=False,
    help="Comma-separated keywords to filter awards.",
)
@click.option(
    "--lookup-path",
    default="data/table/lookup/lookup.M.csv",
    show_default=True,
    help="Path to lookup.M.csv — top concepts used as search keywords.",
)
@click.option(
    "--min-concept-count",
    default=20,
    show_default=True,
    type=int,
    help="Use only lookup concepts with count >= this value as keywords.",
)
@click.option(
    "--date-start",
    default="2015-01-01",
    show_default=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--date-end",
    default="2025-12-31",
    show_default=True,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--out",
    default="data/table/federal-materials.works.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--fetch-limit",
    default=None,
    type=int,
    help="Max awards per agency (None = all).",
)
@click.option(
    "--cache-dir",
    default="/tmp/materials_concepts/.cache/usaspending/",
    show_default=True,
)
def download_usaspending(
    agencies: str,
    keywords: str,
    lookup_path: str,
    min_concept_count: int,
    date_start: str,
    date_end: str,
    out: str,
    fetch_limit: int | None,
    cache_dir: str,
):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # build keyword list from lookup.M.csv + manual keywords
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if lookup_path and Path(lookup_path).exists():
        lookup_df = pd.read_csv(lookup_path)
        if "count" in lookup_df.columns:
            lookup_df = lookup_df[lookup_df["count"] >= min_concept_count]
        lookup_keywords = [
            c.strip() for c in lookup_df["concept"].dropna().tolist()
            if len(c.strip()) >= 8
        ]
        keyword_list = list(set(keyword_list + lookup_keywords))
        logger.info(f"Using {len(keyword_list)} keywords total (lookup + defaults)")

    agency_list = [a.strip().upper() for a in agencies.split(",")]
    all_dfs = []

    for agency_key in agency_list:
        if agency_key not in AGENCIES:
            logger.warning(f"Unknown agency '{agency_key}' — skipping. Available: {list(AGENCIES.keys())}")
            continue

        cache_file = cache_path / f"{agency_key}_{date_start}_{date_end}.csv"
        if cache_file.exists():
            logger.info(f"{agency_key}: loading from cache")
            df = pd.read_csv(cache_file)
        else:
            logger.info(f"{agency_key}: fetching awards from USASpending.gov...")
            records = fetch_agency_awards(
                agency_key, AGENCIES[agency_key], keyword_list,
                date_start, date_end, fetch_limit
            )
            if not records:
                logger.warning(f"{agency_key}: no awards found")
                continue
            df = to_pipeline_format(records, agency_key)
            df.to_csv(cache_file, index=False)
            logger.info(f"{agency_key}: {len(df)} awards saved to cache")

        all_dfs.append(df)

    if not all_dfs:
        logger.error("No awards downloaded.")
        return

    merged = pd.concat(all_dfs).drop_duplicates(subset="id").reset_index(drop=True)
    merged = merged[merged["abstract"].notna() & (merged["abstract"].str.len() >= 100)]
    merged.to_csv(out, index=False)

    logger.info(f"Saved {len(merged)} unique awards to '{out}'")
    logger.info(f"By agency:\n{merged['agency'].value_counts().to_string()}")


if __name__ == "__main__":
    download_usaspending()
