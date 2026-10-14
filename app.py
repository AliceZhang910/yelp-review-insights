"""
Yelp Review Insight Assistant
ISOM5240 practice project

Two Hugging Face pipelines:
  1. Fine-tuned DistilBERT  -> predicts the star rating (1 to 5) of a review
  2. Zero-shot classifier    -> tags what the review is about (food, service, price...)
The app combines both to tell a restaurant owner which reviews need action first.
"""

import os

import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------------
# Settings: change RATING_MODEL to YOUR fine-tuned model on Hugging Face
# ---------------------------------------------------------------
# (The environment variables are only used to test the app locally. On Streamlit Cloud the defaults apply.)
RATING_MODEL = os.environ.get("RATING_MODEL", "Alicyyy/yelp-distilbert-star-rating")
TOPIC_MODEL = os.environ.get("TOPIC_MODEL", "cross-encoder/nli-distilroberta-base")
TOPIC_TEMPLATE = "This review is about {}."   # sentence the zero-shot model checks for each topic (chosen in Notebook 2)

TOPICS = [
    "food and drinks",
    "staff and service",
    "price and value",
    "atmosphere and cleanliness",
    "wait time",
]

# What the owner should do when a negative review is about each topic
ACTIONS = {
    "food and drinks": "Share with the head chef and check the dish or recipe.",
    "staff and service": "Follow up with the shift manager and review staff training.",
    "price and value": "Review portion size and pricing, or offer a loyalty discount.",
    "atmosphere and cleanliness": "Schedule a cleaning and facilities check.",
    "wait time": "Review staffing levels at peak hours.",
}

SAMPLE_REVIEWS = [
    "The ramen was rich and flavorful, and the staff were so friendly. Will be back!",
    "We waited 50 minutes for a table even with a reservation. Food was cold when it arrived.",
    "Decent burger but $28 for a burger and fries is way too much.",
    "The tables were sticky and the bathroom was dirty. Not coming back.",
]

MAX_BATCH_ROWS = 200


# ---------------------------------------------------------------
# Load models once and cache them (so the app stays fast)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models, this takes a minute the first time...")
def load_pipelines():
    rating_pipe = pipeline("text-classification", model=RATING_MODEL)
    topic_pipe = pipeline("zero-shot-classification", model=TOPIC_MODEL)
    return rating_pipe, topic_pipe


def label_to_stars(label: str) -> int:
    """Turn a model label such as '4 stars' or 'LABEL_3' into a number 1 to 5."""
    if label.startswith("LABEL_"):
        return int(label.split("_")[1]) + 1
    return int(label.strip()[0])


def priority_of(stars: int) -> str:
    if stars <= 2:
        return "High: complaint"
    if stars == 3:
        return "Medium: mixed"
    return "Low: praise"


def analyze(texts, rating_pipe, topic_pipe):
    """Run both pipelines on a list of reviews and return a DataFrame."""
    ratings = rating_pipe(texts, truncation=True, max_length=512, batch_size=16)
    topics = topic_pipe(texts, candidate_labels=TOPICS, hypothesis_template=TOPIC_TEMPLATE, batch_size=16)
    if isinstance(topics, dict):  # single review returns a dict, not a list
        topics = [topics]

    rows = []
    for text, r, t in zip(texts, ratings, topics):
        stars = label_to_stars(r["label"])
        main_topic = t["labels"][0]
        rows.append(
            {
                "review": text,
                "predicted_stars": stars,
                "rating_confidence": round(r["score"], 3),
                "main_topic": main_topic,
                "topic_confidence": round(t["scores"][0], 3),
                "priority": priority_of(stars),
                "suggested_action": ACTIONS[main_topic] if stars <= 2 else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------
st.set_page_config(page_title="Review Insight Assistant", page_icon="⭐", layout="wide")
st.title("⭐ Restaurant Review Insight Assistant")
st.caption(
    "Paste a customer review or upload many at once. The app predicts the star rating, "
    "finds what the review is about, and flags complaints that need action."
)

try:
    rating_pipe, topic_pipe = load_pipelines()
except Exception as e:  # clear message instead of a crash
    st.error(f"Could not load the models. Check the model names in app.py. Details: {e}")
    st.stop()

tab_single, tab_batch = st.tabs(["Single review", "Batch upload (CSV)"])

# ---------------- Single review ----------------
with tab_single:
    choice = st.selectbox("Try a sample review, or write your own:", ["(write my own)"] + SAMPLE_REVIEWS)
    default_text = "" if choice == "(write my own)" else choice
    review = st.text_area("Customer review", value=default_text, height=140)

    if st.button("Analyze review", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze([review.strip()], rating_pipe, topic_pipe).iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted rating", "⭐" * int(result["predicted_stars"]),
                      f"{result['rating_confidence']:.0%} confident", delta_color="off")
            c2.metric("Main topic", result["main_topic"].title(),
                      f"{result['topic_confidence']:.0%} confident", delta_color="off")
            c3.metric("Priority", result["priority"])

            if result["suggested_action"]:
                st.error(f"**Suggested action:** {result['suggested_action']}")
            elif result["predicted_stars"] >= 4:
                st.success("Happy customer. Consider thanking them or sharing the review.")
            else:
                st.info("Mixed feedback. Worth a quick read by the manager.")

# ---------------- Batch upload ----------------
with tab_batch:
    st.write(f"Upload a CSV file with a column named **text** (up to {MAX_BATCH_ROWS} reviews).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        if "text" not in df_in.columns:
            st.error("The CSV needs a column named 'text'.")
            st.stop()

        texts = df_in["text"].dropna().astype(str).str.strip()
        texts = texts[texts != ""].tolist()[:MAX_BATCH_ROWS]
        if not texts:
            st.warning("No reviews found in the 'text' column.")
            st.stop()

        with st.spinner(f"Analyzing {len(texts)} reviews..."):
            results = analyze(texts, rating_pipe, topic_pipe)

        complaints = results[results["predicted_stars"] <= 2]
        m1, m2, m3 = st.columns(3)
        m1.metric("Reviews analyzed", len(results))
        m2.metric("Average predicted rating", f"{results['predicted_stars'].mean():.2f} ⭐")
        m3.metric("Complaints to handle", len(complaints))

        left, right = st.columns(2)
        with left:
            st.subheader("Star rating distribution")
            dist = results["predicted_stars"].value_counts().reindex(range(1, 6), fill_value=0)
            dist.index = [str(i) for i in dist.index]  # short labels (1 to 5) so they stay readable
            st.bar_chart(dist, x_label="Predicted stars", y_label="Reviews")
        with right:
            st.subheader("What are complaints about?")
            if complaints.empty:
                st.write("No complaints found. Nice!")
            else:
                st.bar_chart(complaints["main_topic"].value_counts(), horizontal=True,
                             x_label="Topic", y_label="Complaints")

        st.subheader("All reviews (complaints first)")
        st.dataframe(results.sort_values("predicted_stars"))

        st.download_button(
            "Download results as CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="review_insights.csv",
            mime="text/csv",
        )
