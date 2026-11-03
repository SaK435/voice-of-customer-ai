"""Analysis tools the review agent can call. Each one returns a JSON string."""

import json
from pathlib import Path

import pandas as pd

from config import ASPECTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REVIEWS = pd.read_csv(DATA_DIR / "reviews_enriched.csv")
ASPECT_ROWS = pd.read_csv(DATA_DIR / "aspects.csv").merge(
    REVIEWS[["review_id", "title", "text"]], on="review_id", how="left"
)
SENTIMENTS = ["positive", "negative", "neutral"]


def _check_aspect(aspect: str):
    """Return an error message the model can learn from, or None if the aspect is valid."""
    if aspect not in ASPECTS:
        return json.dumps({"error": f"Unknown aspect '{aspect}'. Valid aspects: {ASPECTS}"})
    return None


def list_aspects() -> str:
    """List every aspect with how often customers mention it, split by sentiment.

    Returns:
        JSON with mention counts per aspect and sentiment, most mentioned first.
    """
    table = ASPECT_ROWS.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)
    return table.to_json(orient="index")


def aspect_summary(aspect: str) -> str:
    """Summarise what customers say about one aspect.

    Args:
        aspect: The aspect name, for example "Scent" or "Packaging".

    Returns:
        JSON with number of mentions, share negative and positive, net sentiment
        and the average star rating of reviews that mention it.
    """
    error = _check_aspect(aspect)
    if error:
        return error

    rows = ASPECT_ROWS[ASPECT_ROWS["aspect"] == aspect]
    if rows.empty:
        return json.dumps({"aspect": aspect, "mentions": 0})

    shares = rows["sentiment"].value_counts(normalize=True)
    pos, neg = float(shares.get("positive", 0)), float(shares.get("negative", 0))
    return json.dumps({
        "aspect": aspect,
        "mentions": int(len(rows)),
        "share_positive": round(pos, 2),
        "share_negative": round(neg, 2),
        "net_sentiment": round(pos - neg, 2),
        "avg_rating_when_mentioned": round(float(rows["rating"].mean()), 2),
    })


def low_rating_drivers(top_n: int = 5) -> str:
    """Find which aspects are most often criticised in 1 and 2 star reviews.

    Args:
        top_n: How many aspects to return.

    Returns:
        JSON with the number of low rated reviews and, per aspect, the share of
        those reviews that mention it negatively.
    """
    low_ids = REVIEWS.loc[REVIEWS["rating"] <= 2, "review_id"]
    low_neg = ASPECT_ROWS[
        ASPECT_ROWS["review_id"].isin(low_ids) & (ASPECT_ROWS["sentiment"] == "negative")
    ]
    shares = (
        low_neg.groupby("aspect")["review_id"].nunique() / len(low_ids)
    ).sort_values(ascending=False).head(int(top_n))
    return json.dumps({
        "low_rated_reviews": int(len(low_ids)),
        "share_with_negative_mention": shares.round(2).to_dict(),
    })


def get_evidence(aspect: str, sentiment: str = "negative", n: int = 5) -> str:
    """Get real customer quotes about one aspect, to support an answer with evidence.

    Args:
        aspect: The aspect name, for example "Scent" or "Packaging".
        sentiment: One of "positive", "negative" or "neutral".
        n: How many quotes to return.

    Returns:
        JSON list of quotes with star rating, evidence phrase and the start of the review.
    """
    error = _check_aspect(aspect)
    if error:
        return error
    if sentiment not in SENTIMENTS:
        return json.dumps({"error": f"Unknown sentiment '{sentiment}'. Use one of {SENTIMENTS}"})

    rows = ASPECT_ROWS[(ASPECT_ROWS["aspect"] == aspect) & (ASPECT_ROWS["sentiment"] == sentiment)]
    sample = rows.sample(min(int(n), len(rows)), random_state=0)
    return json.dumps([
        {"rating": row["rating"], "evidence": row["evidence"], "review": str(row["text"])[:200]}
        for _, row in sample.iterrows()
    ])


if __name__ == "__main__":
    print(list_aspects(), "\n")
    print(aspect_summary("Scent"), "\n")
    print(aspect_summary("Smell"), "\n")   # should return a helpful error
    print(low_rating_drivers(3), "\n")
    print(get_evidence("Packaging", "negative", 3))