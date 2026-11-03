"""Run LLM extraction on all reviews, with checkpointing and resume."""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extract import extract_review

RAW_PATH = Path("data/raw/reviews.csv")
OUT_DIR = Path("data/processed")
RESULTS_PATH = OUT_DIR / "extractions.jsonl"
ERRORS_PATH = OUT_DIR / "errors.jsonl"


def load_done_ids() -> set[int]:
    """Review ids that were already processed in an earlier run."""
    if not RESULTS_PATH.exists():
        return set()
    with RESULTS_PATH.open() as f:
        return {json.loads(line)["review_id"] for line in f}


def run(limit: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    if limit:
        df = df.head(limit)

    done_ids = load_done_ids()
    todo = df[~df["review_id"].isin(done_ids)]
    print(f"{len(done_ids)} already done, {len(todo)} to go")

    with RESULTS_PATH.open("a") as results, ERRORS_PATH.open("a") as errors:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Extracting"):
            try:
                result = extract_review(str(row["title"]), str(row["text"]))
                record = {"review_id": int(row["review_id"]), **result.model_dump()}
                results.write(json.dumps(record) + "\n")
                results.flush()
            except Exception as e:
                errors.write(json.dumps({"review_id": int(row["review_id"]), "error": str(e)}) + "\n")
                errors.flush()


def build_tables() -> None:
    """Turn the raw JSON lines into two analysis-ready CSV files."""
    df = pd.read_csv(RAW_PATH)
    with RESULTS_PATH.open() as f:
        records = [json.loads(line) for line in f]

    # One row per review
    review_level = pd.DataFrame([
        {
            "review_id": r["review_id"],
            "overall_sentiment": r["overall_sentiment"],
            "main_complaint": r["main_complaint"],
            "n_aspects": len(r["aspects"]),
        }
        for r in records
    ])
    reviews = df.merge(review_level, on="review_id", how="inner")
    reviews.to_csv(OUT_DIR / "reviews_enriched.csv", index=False)

    # One row per aspect mention
    aspects = pd.DataFrame([
        {"review_id": r["review_id"], **a}
        for r in records
        for a in r["aspects"]
    ])
    aspects = aspects.merge(df[["review_id", "rating", "date"]], on="review_id", how="left")
    aspects.to_csv(OUT_DIR / "aspects.csv", index=False)

    print(f"\nSaved {len(reviews)} reviews and {len(aspects)} aspect mentions")
    print("\nTop aspects:")
    print(aspects["aspect"].value_counts().head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N reviews")
    args = parser.parse_args()

    run(args.limit)
    build_tables()