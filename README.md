# Restaurant Review Insight Assistant

Streamlit app for the ISOM5240 group project. It reads Yelp-style restaurant reviews with two Hugging Face pipelines:

1. **text-classification**: our fine-tuned DistilBERT predicts the star rating (1 to 5)
2. **zero-shot-classification**: a pre-trained NLI model tags the main topic (food, service, price, atmosphere, wait time)

Reviews with 1 or 2 stars are flagged as complaints, and the topic selects a suggested action.

Files: `app.py` (the app), `requirements.txt` (pinned libraries), `topic_eval.csv` and `topic_test.csv` (labelled excerpts used by the experiment notebooks).

Before deploying, set `RATING_MODEL` at the top of `app.py` to your own Hugging Face model name.
