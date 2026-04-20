"""
Extract materials science concepts from grant abstracts using a local Ollama model.
Outputs a CSV with a `llama_concepts` column compatible with the existing pipeline.
"""
import json
import time
from pathlib import Path

import click
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are a materials science expert. Extract key technical concepts from the research abstract below.

Return ONLY a Python list of concepts, like: ['concept one', 'concept two', 'concept three']
- Include materials, processes, techniques, properties, and applications
- Use 2-5 words per concept (no single words, no full sentences)
- Maximum 15 concepts
- No explanations, just the list

Abstract:
{abstract}

Concepts:"""


def call_ollama(abstract: str, model: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": PROMPT_TEMPLATE.format(abstract=abstract[:2000]),  # truncate very long abstracts
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature for consistent extraction
            "num_predict": 200,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return ""


def parse_concepts(raw: str) -> list[str]:
    """Parse LLM output into a list of concept strings."""
    import re
    # find content between [ and ]
    match = re.search(r"\[([^\]]+)\]", raw)
    if not match:
        return []
    inner = match.group(1)
    # split by comma, strip quotes and whitespace
    concepts = [
        c.strip().strip("'\"").strip()
        for c in inner.split(",")
        if c.strip().strip("'\"").strip()
    ]
    return [c.lower() for c in concepts if 4 <= len(c) <= 100]


@click.command("Extract materials science concepts from abstracts using Ollama.")
@click.argument("input_csv")
@click.option("--output", default=None, help="Output CSV path. Defaults to input with .llama.works.csv suffix.")
@click.option("--model", default="llama3:latest", show_default=True, help="Ollama model name.")
@click.option("--batch-size", default=50, show_default=True, type=int, help="Save checkpoint every N rows.")
@click.option("--abstract-col", default="abstract", show_default=True)
@click.option("--id-col", default="id", show_default=True)
@click.option("--start", default=0, type=int, help="Resume from row N (for restarts).")
def extract_concepts_ollama(
    input_csv: str,
    output: str | None,
    model: str,
    batch_size: int,
    abstract_col: str,
    id_col: str,
    start: int,
):
    input_path = Path(input_csv)
    if output is None:
        output = str(input_path.parent / (input_path.stem + ".llama.works.csv"))
    output_path = Path(output)

    # verify Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        logger.info(f"Ollama models available: {available}")
        if model not in available:
            logger.error(f"Model '{model}' not found. Available: {available}")
            return
    except Exception:
        logger.error("Ollama not running. Start it with: ollama serve")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} records from '{input_csv}'")

    # load checkpoint if exists
    if output_path.exists() and start == 0:
        done = pd.read_csv(output_path)
        start = len(done)
        logger.info(f"Resuming from row {start} (checkpoint found)")

    results = []
    for i, row in tqdm(df.iloc[start:].iterrows(), total=len(df) - start, desc="Extracting concepts"):
        abstract = str(row.get(abstract_col, "") or "")
        if len(abstract.strip()) < 50:
            concepts = []
        else:
            raw = call_ollama(abstract, model)
            concepts = parse_concepts(raw)

        results.append({
            "id":             row.get(id_col, ""),
            "llama_concepts": str(concepts),  # store as string repr of list
        })

        # save checkpoint
        if len(results) % batch_size == 0:
            _save(output_path, results, start, df, id_col)
            logger.info(f"  Checkpoint saved at row {start + len(results)}")

    _save(output_path, results, start, df, id_col)
    logger.info(f"Done. Saved {len(results)} records to '{output_path}'")
    logger.info(f"Sample concepts: {results[0]['llama_concepts'] if results else 'none'}")


def _save(output_path: Path, results: list, start: int, df: pd.DataFrame, id_col: str):
    new_df = pd.DataFrame(results)
    if output_path.exists() and start > 0:
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    else:
        combined = new_df
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    extract_concepts_ollama()
