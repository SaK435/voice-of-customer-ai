"""Voice of Customer dashboard: aspects, sentiment and complaints from product reviews."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SENTIMENT_COLOURS = {"positive": "#2E9E6B", "neutral": "#A0A4AB", "negative": "#D9534F"}
MIN_MENTIONS = 20  # hide aspects with too few mentions to be reliable

st.set_page_config(page_title="Voice of Customer AI", layout="wide")


@st.cache_data
def load_data():
    reviews = pd.read_csv(DATA_DIR / "reviews_enriched.csv", parse_dates=["date"])
    aspects = pd.read_csv(DATA_DIR / "aspects.csv", parse_dates=["date"])
    reviews["year"] = reviews["date"].dt.year
    return reviews, aspects


reviews, aspects = load_data()

# Sidebar filters
st.sidebar.header("Filters")
min_year, max_year = int(reviews["year"].min()), int(reviews["year"].max())
year_range = st.sidebar.slider(
    "Review year", min_year, max_year, (max(min_year, 2015), max_year)
)
ratings = st.sidebar.multiselect("Star rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

r = reviews[reviews["year"].between(*year_range) & reviews["rating"].isin(ratings)]
a = aspects[aspects["review_id"].isin(r["review_id"])]

if r.empty or a.empty:
    st.warning("No reviews match these filters.")
    st.stop()

# Header and KPIs
st.title("Voice of Customer AI")
st.caption("Aspects, sentiment and complaint themes extracted from Amazon beauty reviews with a local LLM.")

neg_aspects = a[a["sentiment"] == "negative"]
top_pain = neg_aspects["aspect"].value_counts().idxmax() if not neg_aspects.empty else "None"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", f"{len(r):,}")
c2.metric("Negative reviews", f"{(r['overall_sentiment'] == 'negative').mean():.0%}")
c3.metric("Aspects per review", f"{len(a) / len(r):.1f}")
c4.metric("Top pain point", top_pain)

# Chart 1: mentions by aspect and sentiment
st.subheader("What do customers talk about?")
order = a["aspect"].value_counts().index.tolist()
counts = a.groupby(["aspect", "sentiment"]).size().reset_index(name="mentions")
fig1 = px.bar(
    counts, x="mentions", y="aspect", color="sentiment", orientation="h",
    color_discrete_map=SENTIMENT_COLOURS, category_orders={"aspect": order},
)
fig1.update_yaxes(title=None)
st.plotly_chart(fig1)

left, right = st.columns(2)

# Chart 2: net sentiment per aspect
with left:
    st.subheader("Net sentiment by aspect")
    st.caption(f"(positive minus negative) / all mentions. Aspects with at least {MIN_MENTIONS} mentions.")
    pivot = a.pivot_table(index="aspect", columns="sentiment", values="review_id",
                          aggfunc="count", fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot:
            pivot[col] = 0
    pivot["total"] = pivot[["positive", "neutral", "negative"]].sum(axis=1)
    pivot = pivot[pivot["total"] >= MIN_MENTIONS]
    pivot["net_sentiment"] = (pivot["positive"] - pivot["negative"]) / pivot["total"]
    net = pivot["net_sentiment"].sort_values().reset_index()
    net["direction"] = net["net_sentiment"].apply(lambda x: "positive" if x >= 0 else "negative")
    fig2 = px.bar(net, x="net_sentiment", y="aspect", orientation="h",
                  color="direction", color_discrete_map=SENTIMENT_COLOURS)
    fig2.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_range=[-1, 1])
    fig2.update_yaxes(title=None)
    st.plotly_chart(fig2)

# Chart 3: what drives low ratings
with right:
    st.subheader("What drives 1 and 2 star reviews?")
    low_ids = r.loc[r["rating"] <= 2, "review_id"]
    if len(low_ids) == 0:
        st.info("No 1 or 2 star reviews in the current filter.")
    else:
        st.caption(f"Share of {len(low_ids)} low rated reviews with a negative mention of each aspect.")
        low_neg = a[a["review_id"].isin(low_ids) & (a["sentiment"] == "negative")]
        drivers = (
            low_neg.groupby("aspect")["review_id"].nunique() / len(low_ids)
        ).sort_values().reset_index(name="share")
        fig3 = px.bar(drivers, x="share", y="aspect", orientation="h",
                      color_discrete_sequence=[SENTIMENT_COLOURS["negative"]])
        fig3.update_layout(xaxis_tickformat=".0%")
        fig3.update_yaxes(title=None)
        st.plotly_chart(fig3)

# Evidence explorer
st.subheader("Evidence explorer")
col_a, col_b = st.columns([2, 3])
selected_aspect = col_a.selectbox("Aspect", order)
selected_sentiment = col_b.radio("Sentiment", ["negative", "positive", "neutral"], horizontal=True)

evidence = (
    a[(a["aspect"] == selected_aspect) & (a["sentiment"] == selected_sentiment)]
    .merge(r[["review_id", "title", "text"]], on="review_id")
)
st.write(f"{len(evidence)} mentions")
st.dataframe(evidence[["rating", "evidence", "title", "text"]], hide_index=True)