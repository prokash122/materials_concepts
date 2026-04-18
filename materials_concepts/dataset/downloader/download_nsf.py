import time
from pathlib import Path

import click
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

NSF_API_URL = "https://api.nsf.gov/services/v1/awards.json"

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


def load_keywords_from_lookup(lookup_path: str, min_count: int, in_graph_only: bool) -> list[str]:
    df = pd.read_csv(lookup_path)
    if in_graph_only and "in_graph" in df.columns:
        df = df[df["in_graph"].astype(str).str.upper() == "TRUE"]
    if "count" in df.columns:
        df = df[df["count"] >= min_count]
    keywords = df["concept"].dropna().str.strip().tolist()
    logger.info(f"Loaded {len(keywords)} keywords from '{lookup_path}' (min_count={min_count})")
    return keywords


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

    with tqdm(desc=f"  '{keyword}'", unit=" awards", leave=False) as pbar:
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
    renamed["publication_date"] = pd.to_datetime(
        renamed["publication_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return renamed


@click.command("Download NSF awards using concepts from lookup.M.csv as keywords.")
@click.option(
    "--lookup-path",
    default="data/table/lookup/lookup.M.csv",
    show_default=True,
    help="Path to lookup.M.csv — concepts will be used as NSF search keywords.",
)
@click.option(
    "--min-count",
    default=10,
    show_default=True,
    type=int,
    help="Only use concepts with count >= this value (avoids rare/noisy concepts).",
)
@click.option(
    "--in-graph-only",
    default=True,
    show_default=True,
    type=bool,
    help="Only use concepts where in_graph=True.",
)
@click.option(
    "--keywords",
    default=None,
    help="Override: comma-separated keywords instead of lookup file.",
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
    lookup_path: str,
    min_count: int,
    in_graph_only: bool,
    keywords: str | None,
    date_start: str,
    date_end: str,
    out: str,
    fetch_limit: int | None,
    cache_dir: str,
):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # load keywords from lookup.M.csv or manual override
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",")]
        logger.info(f"Using {len(keyword_list)} manually provided keywords")
    else:
        keyword_list = load_keywords_from_lookup(lookup_path, min_count, in_graph_only)

    all_dfs = []
    total = len(keyword_list)

    for i, keyword in enumerate(keyword_list, 1):
        cache_file = cache_path / f"{keyword.replace(' ', '_').replace('/', '-')}.csv"

        if cache_file.exists():
            logger.info(f"({i}/{total}) Cached: '{keyword}'")
            df = pd.read_csv(cache_file)
        else:
            logger.info(f"({i}/{total}) Fetching: '{keyword}'")
            awards = fetch_awards(keyword, date_start, date_end, fetch_limit)

            if not awards:
                logger.debug(f"No awards found for '{keyword}'")
                # write empty cache to skip on re-run
                pd.DataFrame().to_csv(cache_file, index=False)
                continue

            df = to_pipeline_format(pd.DataFrame(awards))
            df.to_csv(cache_file, index=False)
            logger.info(f"  → {len(df)} awards found")

        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No awards downloaded. Check your lookup file or date range.")
        return

    merged = pd.concat(all_dfs).drop_duplicates(subset="id").reset_index(drop=True)
    merged = merged[merged["abstract"].notna() & (merged["abstract"].str.strip() != "")]
    merged.to_csv(out, index=False)

    logger.info(f"Done! Saved {len(merged)} unique awards to '{out}'")
    logger.info(f"Date range: {date_start} → {date_end}")
    logger.info(f"Keywords searched: {total}, with results: {len(all_dfs)}")


if __name__ == "__main__":
    download_nsf()
