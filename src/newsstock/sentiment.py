"""Headline sentiment scoring with FinBERT and per-session aggregation.

:class:`FinBertScorer` uses the finance-specific ProsusAI/finbert model from
Hugging Face. :class:`LexiconScorer` is a lightweight fallback (a small
finance word list) used for offline tests or when ``transformers`` is not
installed; results from it should not be reported as FinBERT results.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LABELS = ["positive", "negative", "neutral"]


class FinBertScorer:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.order = [next(i for i, l in id2label.items() if l == lab) for lab in LABELS]

    def score(self, texts: list[str]) -> np.ndarray:
        """Return an (n, 3) array of [positive, negative, neutral] probabilities."""
        probs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                p = self.torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy()
                probs.append(p[:, self.order])
        return np.vstack(probs) if probs else np.zeros((0, 3))


class LexiconScorer:
    name = "lexicon"

    POSITIVE = {
        "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains", "jump", "jumps",
        "record", "growth", "upgrade", "upgraded", "strong", "profit", "profits", "rise", "rises", "outperform",
        "bullish", "boost", "boosts", "approval", "wins", "expands", "raises", "higher", "tops",
    }
    NEGATIVE = {
        "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops", "slump", "slumps", "loss", "losses",
        "downgrade", "downgraded", "weak", "lawsuit", "probe", "recall", "cut", "cuts", "decline", "declines",
        "bearish", "layoffs", "fine", "fined", "warns", "lower", "sinks", "crash", "investigation",
    }

    def score(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z]+", str(t).lower())
            pos = sum(w in self.POSITIVE for w in words)
            neg = sum(w in self.NEGATIVE for w in words)
            logits = np.array([pos, neg, 0.5])
            e = np.exp(logits - logits.max())
            out[i] = e / e.sum()
        return out


def get_scorer(prefer_finbert: bool = True, model_name: str = "ProsusAI/finbert"):
    if prefer_finbert:
        try:
            return FinBertScorer(model_name)
        except Exception as exc:  # missing packages or no network for the model download
            print(f"[sentiment] FinBERT unavailable ({exc!r}); falling back to lexicon scorer.")
    return LexiconScorer()


def score_headlines(news: pd.DataFrame, scorer) -> pd.DataFrame:
    probs = scorer.score(news["headline"].astype(str).tolist())
    out = news.copy()
    out[["p_pos", "p_neg", "p_neu"]] = probs
    out["sent_label"] = np.array(LABELS)[probs.argmax(axis=1)] if len(out) else []
    return out


def aggregate_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored headlines to one row per (ticker, session)."""
    s = scored.assign(
        net=scored["p_pos"] - scored["p_neg"],
        is_pos=(scored["sent_label"] == "positive").astype(float),
        is_neg=(scored["sent_label"] == "negative").astype(float),
    )
    agg = s.groupby(["ticker", "session"]).agg(
        news_count=("headline", "size"),
        sent_pos_mean=("p_pos", "mean"),
        sent_neg_mean=("p_neg", "mean"),
        sent_neu_mean=("p_neu", "mean"),
        sent_net_mean=("net", "mean"),
        sent_net_std=("net", "std"),
        sent_pos_share=("is_pos", "mean"),
        sent_neg_share=("is_neg", "mean"),
    ).reset_index()
    agg["sent_net_std"] = agg["sent_net_std"].fillna(0.0)
    return agg.rename(columns={"session": "date"})
