"""Print random evidence phrases per aspect to check label quality."""

import pandas as pd

aspects = pd.read_csv("data/processed/aspects.csv")

for aspect in ["Ease of use", "Authenticity", "Other"]:
    subset = aspects[aspects["aspect"] == aspect]
    print("=" * 60)
    print(f"{aspect} ({len(subset)} mentions)")
    print("=" * 60)
    for _, row in subset.sample(15, random_state=0).iterrows():
        print(f"[{row['sentiment']}] {row['evidence']}")
    print()