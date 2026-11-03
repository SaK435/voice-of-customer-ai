"""Extract aspects, sentiment and main complaint from a review using a local LLM."""

import re
from typing import Literal, Optional

import ollama
from pydantic import BaseModel

from config import ASPECT_DEFINITIONS, ASPECTS, MODEL_NAME

AspectName = Literal[tuple(ASPECTS)]


class AspectMention(BaseModel):
    aspect: AspectName
    sentiment: Literal["positive", "negative", "neutral"]
    evidence: str


class ReviewExtraction(BaseModel):
    overall_sentiment: Literal["positive", "negative", "mixed", "neutral"]
    aspects: list[AspectMention]
    main_complaint: Optional[str]


ASPECT_GUIDE = "\n".join(f"- {name}: {desc}" for name, desc in ASPECT_DEFINITIONS.items())

SYSTEM_PROMPT = f"""You analyse customer product reviews for a consumer insights team.

Aspect categories (use ONLY these, pick the closest match):
{ASPECT_GUIDE}

Return JSON with:
- overall_sentiment: positive, negative, mixed or neutral, from the reviewer's point of view.
- aspects: the aspects the reviewer clearly talks about. For each give its sentiment and a short evidence phrase copied exactly from the review (max 10 words).
- main_complaint: one short sentence describing what THIS reviewer disliked, or null if they had no complaint.

Rules:
- Each evidence phrase may be used for only ONE aspect.
- Tag as few aspects as needed. Most reviews have 1 to 3 aspects.
- Use "Other" only if nothing else fits.

Example:
Review: "Lovely smell but the pump broke after a week and it was overpriced."
Output: {{"overall_sentiment": "mixed", "aspects": [
  {{"aspect": "Scent", "sentiment": "positive", "evidence": "Lovely smell"}},
  {{"aspect": "Packaging", "sentiment": "negative", "evidence": "the pump broke after a week"}},
  {{"aspect": "Price and value", "sentiment": "negative", "evidence": "it was overpriced"}}],
  "main_complaint": "The pump broke quickly and the product is overpriced."}}"""


def _normalise(text: str) -> str:
    """Lowercase and keep only letters, digits and single spaces."""
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def clean_aspects(result: ReviewExtraction, review_text: str) -> ReviewExtraction:
    """Drop aspects whose evidence is not in the review, and drop reused evidence."""
    review_norm = _normalise(review_text)
    seen_evidence = set()
    kept = []

    for mention in result.aspects:
        evidence_norm = _normalise(mention.evidence)
        if not evidence_norm:
            continue

        # Grounding check: the first few words of the evidence must appear in the review
        first_words = " ".join(evidence_norm.split()[:4])
        if first_words not in review_norm:
            continue

        # No reused evidence across aspects
        if evidence_norm in seen_evidence:
            continue

        seen_evidence.add(evidence_norm)
        kept.append(mention)

    result.aspects = kept
    return result


def extract_review(title: str, text: str) -> ReviewExtraction:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\nReview: {text}"},
        ],
        format=ReviewExtraction.model_json_schema(),
        options={"temperature": 0},
    )
    result = ReviewExtraction.model_validate_json(response.message.content)
    return clean_aspects(result, f"{title} {text}")