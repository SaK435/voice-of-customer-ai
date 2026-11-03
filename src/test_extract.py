import time

import pandas as pd

from extract import extract_review

df = pd.read_csv("data/raw/reviews.csv")
sample = df.sample(10, random_state=1)

start = time.time()
for _, row in sample.iterrows():
    result = extract_review(str(row["title"]), str(row["text"]))
    print("=" * 80)
    print(f"Rating: {row['rating']}  |  {row['title']}")
    print(str(row["text"])[:300])
    print("->", result.model_dump_json(indent=2))

print(f"\nAverage time per review: {(time.time() - start) / 10:.1f} s")