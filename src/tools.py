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
def _short(text: str, max_words: int = 12) -> str:
    """Cut long quotes to a readable length."""
    words = str(text).split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

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
        JSON with a ready made headline and, per aspect, the share of low rated
        reviews that mention it negatively.
    """
    low_ids = REVIEWS.loc[REVIEWS["rating"] <= 2, "review_id"]
    low_neg = ASPECT_ROWS[
        ASPECT_ROWS["review_id"].isin(low_ids) & (ASPECT_ROWS["sentiment"] == "negative")
    ]
    shares = (
        low_neg.groupby("aspect")["review_id"].nunique() / len(low_ids)
    ).sort_values(ascending=False).head(int(top_n))

    if shares.empty:
        return json.dumps({"headline": "No negative aspect mentions in low rated reviews."})

    top_aspect, top_share = shares.index[0], shares.iloc[0]
    headline = (
        f"Of {len(low_ids)} reviews with 1 or 2 stars, {top_share:.0%} criticise "
        f"{top_aspect.lower()}, the most common complaint."
    )
    return json.dumps({
        "headline": headline,
        "share_with_negative_mention": {k: f"{v:.0%}" for k, v in shares.items()},
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


def aspect_report(aspect: str) -> str:
    """Full picture of one aspect: how customers feel about it, with example quotes.
    Use this for ANY question about a single aspect, such as scent, price or packaging.

    Args:
        aspect: The aspect name, for example "Scent" or "Packaging".

    Returns:
        JSON with a ready made headline, the key numbers, and 2 positive and
        2 negative customer quotes.
    """
    error = _check_aspect(aspect)
    if error:
        return error

    summary = json.loads(aspect_summary(aspect))
    if summary.get("mentions", 0) == 0:
        return json.dumps({"headline": f"No reviews mention {aspect.lower()}."})

    headline = (
        f"{summary['share_positive']:.0%} of {summary['mentions']} {aspect.lower()} mentions "
        f"are positive and {summary['share_negative']:.0%} negative "
        f"(average rating {summary['avg_rating_when_mentioned']} stars)."
    )
    report = {"headline": headline, **summary}
    for sentiment in ["positive", "negative"]:
        quotes = json.loads(get_evidence(aspect, sentiment, 2))
        report[f"{sentiment}_quotes"] = [
            {"quote": _short(q["evidence"]), "stars": q["rating"]} for q in quotes
        ]
    return json.dumps(report)

if __name__ == "__main__":
    print(list_aspects(), "\n")
    print(aspect_summary("Scent"), "\n")
    print(aspect_summary("Smell"), "\n")   # should return a helpful error
    print(low_rating_drivers(3), "\n")
    print(get_evidence("Packaging", "negative", 3))