"""Scoring headline sentiment with FinBERT and summarizing it per trading day.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text. For
each headline it gives three probabilities: positive, negative and neutral.

If FinBERT can't be loaded (no internet, or transformers isn't installed) we
fall back to a very simple word-counting scorer. That fallback is only meant
for testing the code; real results should come from FinBERT.
"""

import re

import numpy as np
import pandas as pd

LABELS = ["positive", "negative", "neutral"]


class FinBertScorer:
    name = "finbert"

    def __init__(self, model_name="ProsusAI/finbert", batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        # The model's outputs aren't necessarily in our LABELS order,
        # so we look up which output column is which label.
        model_labels = {i: label.lower() for i, label in self.model.config.id2label.items()}
        self.column_order = [next(i for i, l in model_labels.items() if l == wanted) for wanted in LABELS]

    def score(self, texts):
        """Return one row per headline: [p_positive, p_negative, p_neutral]."""
        results = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
                tokens = tokens.to(self.device)
                logits = self.model(**tokens).logits
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
                results.append(probs[:, self.column_order])
        if not results:
            return np.zeros((0, 3))
        return np.vstack(results)


class WordListScorer:
    """Backup scorer: counts positive and negative finance words in a headline."""

    name = "word list (fallback)"

    POSITIVE_WORDS = {
        "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains", "jump", "jumps",
        "record", "growth", "upgrade", "upgraded", "strong", "profit", "profits", "rise", "rises", "outperform",
        "bullish", "boost", "boosts", "approval", "wins", "expands", "raises", "higher", "tops",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops", "slump", "slumps", "loss", "losses",
        "downgrade", "downgraded", "weak", "lawsuit", "probe", "recall", "cut", "cuts", "decline", "declines",
        "bearish", "layoffs", "fine", "fined", "warns", "lower", "sinks", "crash", "investigation",
    }

    def score(self, texts):
        scores = np.zeros((len(texts), 3))
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z]+", str(text).lower())
            n_pos = sum(word in self.POSITIVE_WORDS for word in words)
            n_neg = sum(word in self.NEGATIVE_WORDS for word in words)
            # Turn the counts into probabilities with a softmax.
            # The 0.5 for neutral means a headline with no signal words leans neutral.
            raw = np.array([n_pos, n_neg, 0.5])
            exp = np.exp(raw - raw.max())
            scores[i] = exp / exp.sum()
        return scores


def get_scorer(use_finbert=True, model_name="ProsusAI/finbert"):
    if use_finbert:
        try:
            return FinBertScorer(model_name)
        except Exception as error:
            print(f"Couldn't load FinBERT ({error}). Using the word-list scorer instead.")
    return WordListScorer()


def score_headlines(headlines, scorer):
    probs = scorer.score(headlines["headline"].astype(str).tolist())
    scored = headlines.copy()
    scored["p_pos"] = probs[:, 0]
    scored["p_neg"] = probs[:, 1]
    scored["p_neu"] = probs[:, 2]
    # The label is whichever of the three has the highest probability.
    scored["sent_label"] = [LABELS[i] for i in probs.argmax(axis=1)] if len(scored) else []
    return scored


def aggregate_sentiment(scored):
    """Combine all headlines for the same stock and trading day into one row."""
    scored = scored.copy()
    scored["net"] = scored["p_pos"] - scored["p_neg"]  # above 0 = more positive than negative
    scored["is_pos"] = (scored["sent_label"] == "positive").astype(float)
    scored["is_neg"] = (scored["sent_label"] == "negative").astype(float)

    daily = scored.groupby(["ticker", "session"]).agg(
        news_count=("headline", "size"),
        sent_pos_mean=("p_pos", "mean"),
        sent_neg_mean=("p_neg", "mean"),
        sent_neu_mean=("p_neu", "mean"),
        sent_net_mean=("net", "mean"),
        sent_net_std=("net", "std"),
        sent_pos_share=("is_pos", "mean"),
        sent_neg_share=("is_neg", "mean"),
    ).reset_index()

    # With only one headline the standard deviation is undefined; call it 0.
    daily["sent_net_std"] = daily["sent_net_std"].fillna(0.0)
    return daily.rename(columns={"session": "date"})
