from pathlib import Path

import click
import pandas as pd
from loguru import logger


@click.command("Split assistance grants CSV into train/test sets by year.")
@click.argument("input_csv")
@click.option("--train-end", default=2019, show_default=True, type=int,
              help="Last year (inclusive) for training set.")
@click.option("--test-end", default=2024, show_default=True, type=int,
              help="Last year (inclusive) for test set.")
@click.option("--out-dir", default="data/table", show_default=True,
              help="Output directory for train/test CSVs.")
def split_by_year(input_csv: str, train_end: int, test_end: int, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} records from '{input_csv}'")

    df["year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year
    logger.info(f"Year distribution:\n{df['year'].value_counts().sort_index().to_string()}")

    train = df[df["year"] <= train_end].drop(columns="year").reset_index(drop=True)
    test  = df[(df["year"] > train_end) & (df["year"] <= test_end)].drop(columns="year").reset_index(drop=True)
    pred  = df[df["year"] > test_end].drop(columns="year").reset_index(drop=True)

    train_path = out_path / "assistance-train.works.csv"
    test_path  = out_path / "assistance-test.works.csv"
    pred_path  = out_path / "assistance-predict.works.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    pred.to_csv(pred_path, index=False)

    logger.info(f"Train ({train_end} and earlier): {len(train):,} records → {train_path}")
    logger.info(f"Test  ({train_end+1}–{test_end}):      {len(test):,} records → {test_path}")
    logger.info(f"Predict ({test_end+1}+):              {len(pred):,} records → {pred_path}")


if __name__ == "__main__":
    split_by_year()
