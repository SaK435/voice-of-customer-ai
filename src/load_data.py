"""Stream Amazon reviews and save a small random sample to data/raw/."""

import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/All_Beauty.jsonl"
)
MAX_REVIEWS = 20_000   # how many usable reviews to read from the stream
SAMPLE_SIZE = 2_000    # how many to keep for analysis
MIN_TEXT_LEN = 40      # skip very short reviews like "great!"
OUT_PATH = Path("data/raw/reviews.csv")


def stream_reviews(url: str, max_reviews: int) -> list[dict]:
    """Read reviews line by line and stop once we have enough."""
    rows = []
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        progress = tqdm(total=max_reviews, desc="Reading reviews")
        for line in response.iter_lines():
            if not line:
                continue
            review = json.loads(line)
            text = (review.get("text") or "").strip()
            if len(text) < MIN_TEXT_LEN:
                continue
            rows.append({
                "review_id": len(rows),
                "product_id": review.get("parent_asin"),
                "rating": review.get("rating"),
                "title": review.get("title"),
                "text": text,
                "helpful_votes": review.get("helpful_vote"),
                "verified": review.get("verified_purchase"),
                "timestamp": review.get("timestamp"),
            })
            progress.update(1)
            if len(rows) >= max_reviews:
                break
        progress.close()
    return rows


def main() -> None:
    rows = stream_reviews(URL, MAX_REVIEWS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df = df.drop(columns="timestamp")

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(sample)} reviews to {OUT_PATH}")
    print("\nRating distribution:")
    print(sample["rating"].value_counts().sort_index())
    print(f"\nUnique products: {sample['product_id'].nunique()}")
    print(f"Date range: {sample['date'].min()} to {sample['date'].max()}")


if __name__ == "__main__":
    main()