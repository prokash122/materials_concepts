"""
Extract materials science concepts from grant abstracts using the Claude Batch API.
- 50% cheaper than standard API calls
- Prompt caching on system prompt (saves ~90% on repeated tokens)
- Saves llama_concepts column compatible with the existing pipeline
"""
import json
import re
import time
from pathlib import Path

import anthropic
import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

SYSTEM_PROMPT = """You are a materials science expert. Extract key technical concepts from research grant abstracts.

Rules:
- Return ONLY a Python list like: ['concept one', 'concept two', 'concept three']
- Include: materials, processes, techniques, properties, and applications
- Each concept must be 2-5 words (no single words, no full sentences)
- Maximum 15 concepts per abstract
- No explanations — just the list"""


def make_batch_request(row_id: str, abstract: str) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    return {
        "custom_id": str(row_id),
        "params": MessageCreateParamsNonStreaming(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache system prompt across all requests
            }],
            messages=[{
                "role": "user",
                "content": f"Extract materials science concepts from this grant abstract:\n\n{abstract[:2000]}"
            }],
        ),
    }


def parse_concepts(text: str) -> list[str]:
    match = re.search(r"\[([^\]]+)\]", text)
    if not match:
        return []
    inner = match.group(1)
    concepts = [
        c.strip().strip("'\"").strip()
        for c in inner.split(",")
        if c.strip().strip("'\"").strip()
    ]
    return [c.lower() for c in concepts if 4 <= len(c) <= 100]


def poll_until_done(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30) -> None:
    logger.info(f"Polling batch {batch_id} every {poll_interval}s...")
    with tqdm(desc="Waiting for batch", unit=" polls") as pbar:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            pbar.set_postfix(
                processing=counts.processing,
                succeeded=counts.succeeded,
                errored=counts.errored,
            )
            pbar.update(1)
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval)
    logger.info(f"Batch complete — succeeded: {counts.succeeded}, errored: {counts.errored}")


@click.command("Extract concepts from grant abstracts using Claude Batch API.")
@click.argument("input_csv")
@click.option("--output", default=None,
              help="Output CSV path. Defaults to input stem + .claude.works.csv")
@click.option("--abstract-col", default="abstract", show_default=True)
@click.option("--id-col", default="id", show_default=True)
@click.option("--batch-size", default=10000, show_default=True, type=int,
              help="Max rows per batch (API limit: 100,000).")
@click.option("--poll-interval", default=30, show_default=True, type=int,
              help="Seconds between batch status polls.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY",
              help="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
def extract_concepts_claude(
    input_csv: str,
    output: str | None,
    abstract_col: str,
    id_col: str,
    batch_size: int,
    poll_interval: int,
    api_key: str | None,
):
    input_path = Path(input_csv)
    if output is None:
        output = str(input_path.parent / (input_path.stem + ".claude.works.csv"))
    output_path = Path(output)

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df):,} records from '{input_csv}'")

    # skip already processed rows if output exists
    done_ids: set = set()
    if output_path.exists():
        done_df = pd.read_csv(output_path)
        done_ids = set(done_df["id"].astype(str).tolist())
        logger.info(f"Resuming — {len(done_ids):,} rows already done")

    pending = df[~df[id_col].astype(str).isin(done_ids)]
    logger.info(f"{len(pending):,} rows to process")

    if pending.empty:
        logger.info("Nothing to do.")
        return

    all_results: list[dict] = []

    # process in batches
    chunks = [pending.iloc[i:i+batch_size] for i in range(0, len(pending), batch_size)]
    for chunk_idx, chunk in enumerate(chunks, 1):
        logger.info(f"Submitting batch {chunk_idx}/{len(chunks)} ({len(chunk):,} rows)...")

        requests = []
        for _, row in chunk.iterrows():
            abstract = str(row.get(abstract_col, "") or "")
            if len(abstract.strip()) < 50:
                all_results.append({"id": str(row[id_col]), "llama_concepts": "[]"})
                continue
            requests.append(make_batch_request(str(row[id_col]), abstract))

        if not requests:
            continue

        batch = client.messages.batches.create(requests=requests)
        logger.info(f"Batch submitted: {batch.id}")

        poll_until_done(client, batch.id, poll_interval)

        # collect results
        for result in client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                msg = result.result.message
                text = next((b.text for b in msg.content if b.type == "text"), "")
                concepts = parse_concepts(text)
                all_results.append({
                    "id": result.custom_id,
                    "llama_concepts": str(concepts),
                })
            else:
                logger.warning(f"Row {result.custom_id} failed: {result.result.type}")
                all_results.append({"id": result.custom_id, "llama_concepts": "[]"})

        # checkpoint after each batch
        _save(output_path, all_results, done_ids, df, id_col)
        logger.info(f"Checkpoint saved ({len(all_results):,} new rows)")

    _save(output_path, all_results, done_ids, df, id_col)
    logger.info(f"Done. Results saved to '{output_path}'")

    # show sample
    final = pd.read_csv(output_path)
    logger.info(f"Total rows: {len(final):,}")
    if not final.empty:
        logger.info(f"Sample concepts:\n{final['llama_concepts'].iloc[0]}")


def _save(output_path: Path, new_results: list[dict], done_ids: set, df: pd.DataFrame, id_col: str):
    new_df = pd.DataFrame(new_results)
    if output_path.exists() and done_ids:
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    extract_concepts_claude()
