import time
from pathlib import Path

import click
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

NSF_API_URL = "https://api.nsf.gov/services/v1/awards.json"

# NSF programs focused on materials science and manufacturing
NSF_PROGRAMS = [
    "Division of Materials Research",
    "Civil, Mechanical and Manufacturing Innovation",
    "Advanced Manufacturing",
    "Materials Engineering and Processing",
    "Metals, Minerals, and Mining",
    "Ceramics",
    "Polymers",
    "Solid State and Materials Chemistry",
    "Condensed Matter Physics",
    "Biomaterials",
]

FIELDS = ",".join([
    "id",
    "title",
    "abstractText",
    "date",
    "startDate",
    "expDate",
    "fundProgramName",
    "awardeeName",
    "estimatedTotalAmt",
    "piFirstName",
    "piLastName",
    "agency",
])


def fetch_page(keyword: str, offset: int, date_start: str, date_end: str) -> list:
    params = {
        "keyword": keyword,
        "dateStart": date_start,
        "dateEnd": date_end,
        "fields": FIELDS,
        "rpp": 25,
        "offset": offset,
    }
    try:
        response = requests.get(NSF_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", {}).get("award", []) or []
    except Exception as e:
        logger.warning(f"Request failed at offset {offset}: {e}")
        return []


def fetch_awards(keyword: str, date_start: str, date_end: str, fetch_limit: int | None) -> list:
    results = []
    offset = 1

    with tqdm(desc=f"Downloading '{keyword}'", unit=" awards") as pbar:
        while True:
            awards = fetch_page(keyword, offset, date_start, date_end)
            if not awards:
                break

            results.extend(awards)
            pbar.update(len(awards))
            offset += 25

            if fetch_limit and len(results) >= fetch_limit:
                results = results[:fetch_limit]
                break

            time.sleep(0.3)  # be respectful to NSF API

    return results


def to_pipeline_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NSF awards to the same format as OpenAlex works pipeline."""
    renamed = pd.DataFrame({
        "id":               df.get("id", ""),
        "display_name":     df.get("title", ""),
        "abstract":         df.get("abstractText", ""),
        "publication_date": df.get("startDate", ""),
        "program":          df.get("fundProgramName", ""),
        "awardee":          df.get("awardeeName", ""),
        "amount":           df.get("estimatedTotalAmt", ""),
        "pi":               df.get("piFirstName", "") + " " + df.get("piLastName", ""),
        "is_retracted":     False,
        "is_paratext":      False,
    })
    # normalize date to YYYY-MM-DD
    renamed["publication_date"] = pd.to_datetime(
        renamed["publication_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return renamed


@click.command("Download NSF awards for materials science and manufacturing.")
@click.option(
    "--keywords",
    default="materials science,advanced manufacturing,smart materials,piezoelectric,ceramics,polymers,composites",
    help="Comma-separated list of keywords to search for.",
)
@click.option(
    "--date-start",
    default="01/01/2010",
    show_default=True,
    help="Start date for award search (MM/DD/YYYY).",
)
@click.option(
    "--date-end",
    default="12/31/2023",
    show_default=True,
    help="End date for award search (MM/DD/YYYY).",
)
@click.option(
    "--out",
    default="data/table/nsf-materials-manufacturing.works.csv",
    show_default=True,
    help="Output CSV file path.",
)
@click.option(
    "--fetch-limit",
    default=None,
    type=int,
    help="Max awards to fetch per keyword (None = all).",
)
@click.option(
    "--cache-dir",
    default="/tmp/materials_concepts/.cache/nsf/",
    show_default=True,
    help="Directory to cache per-keyword CSV files.",
)
def download_nsf(
    keywords: str,
    date_start: str,
    date_end: str,
    out: str,
    fetch_limit: int | None,
    cache_dir: str,
):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    keyword_list = [k.strip() for k in keywords.split(",")]
    all_dfs = []

    for keyword in keyword_list:
        cache_file = cache_path / f"{keyword.replace(' ', '_')}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached results for '{keyword}'")
            df = pd.read_csv(cache_file)
        else:
            logger.info(f"Fetching awards for keyword: '{keyword}'")
            awards = fetch_awards(keyword, date_start, date_end, fetch_limit)

            if not awards:
                logger.warning(f"No awards found for '{keyword}'")
                continue

            df = to_pipeline_format(pd.DataFrame(awards))
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} awards for '{keyword}'")

        all_dfs.append(df)

    if not all_dfs:
        logger.error("No awards downloaded. Check your keywords or date range.")
        return

    merged = pd.concat(all_dfs).drop_duplicates(subset="id").reset_index(drop=True)
    # remove entries with empty abstracts
    merged = merged[merged["abstract"].notna() & (merged["abstract"].str.strip() != "")]
    merged.to_csv(out, index=False)

    logger.info(f"Saved {len(merged)} unique awards to '{out}'")
    logger.info(f"Date range: {date_start} → {date_end}")
    logger.info(f"Keywords searched: {keyword_list}")


if __name__ == "__main__":
    download_nsf()
