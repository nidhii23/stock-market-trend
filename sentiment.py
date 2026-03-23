"""
sentiment.py — News sentiment using FinBERT + Google News RSS fallback.

Fix: yfinance news broken for NSE stocks → replaced with Google News RSS
     via feedparser, which reliably returns Indian stock headlines.

Engine priority:
  1. FinBERT  (best — finance-specific BERT model)
  2. VADER    (fallback if transformers not installed)
Both now use Google News RSS headlines instead of yfinance.
"""

import logging
import re

import yfinance as yf

logger = logging.getLogger(__name__)

_MAX_ARTICLES = 15
_STRONG_POS   =  0.20
_STRONG_NEG   = -0.20

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        # Import directly from submodule to avoid Streamlit caching issues
        from transformers.pipelines import pipeline as hf_pipeline
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded.")
        return _finbert_pipeline
    except Exception:
        try:
            # Second attempt with standard import
            import transformers
            _finbert_pipeline = transformers.pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded (fallback import).")
            return _finbert_pipeline
        except Exception as exc:
            logger.warning("FinBERT unavailable: %s — using VADER.", exc)
            return None


def _finbert_score(title: str, pipe) -> float:
    try:
        results = pipe(title[:512])[0]
        scores  = {r["label"].lower(): r["score"] for r in results}
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)
    except Exception:
        return 0.0


def _vader_score(title: str) -> float:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer().polarity_scores(title)["compound"]
    except Exception:
        return 0.0


def _fetch_google_news(query: str) -> list[str]:
    """
    Fetch headlines from Google News RSS — works reliably for Indian stocks.
    Returns list of headline strings.
    """
    try:
        import feedparser
        # Clean ticker for search — remove .NS .BO suffix
        clean = re.sub(r"\.(NS|BO|BSE)$", "", query, flags=re.IGNORECASE)
        url   = (
            f"https://news.google.com/rss/search?"
            f"q={clean}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed  = feedparser.parse(url)
        titles = []
        for entry in feed.entries[:_MAX_ARTICLES]:
            title = entry.get("title", "").strip()
            # Remove source suffix like " - Economic Times"
            title = re.sub(r"\s*-\s*[^-]+$", "", title).strip()
            if title:
                titles.append(title)
        logger.info("Google News RSS: %d headlines for %s", len(titles), query)
        return titles
    except Exception as exc:
        logger.warning("Google News RSS failed for %s: %s", query, exc)
        return []


def _fetch_yfinance_news(ticker: str) -> list[str]:
    """Fallback: yfinance headlines (unreliable for NSE)."""
    try:
        news   = yf.Ticker(ticker).get_news() or []
        return [
            (a.get("title") or a.get("headline") or "").strip()
            for a in news[:_MAX_ARTICLES]
            if (a.get("title") or a.get("headline") or "").strip()
        ]
    except Exception:
        return []


def _neutral_result(engine="none") -> dict:
    return {
        "score": 0.0, "normalised": 0.5, "signal": "NEUTRAL",
        "positive_news": [], "negative_news": [], "engine": engine,
    }


def get_news_sentiment(ticker: str) -> dict:
    """
    Fetch headlines via Google News RSS (primary) or yfinance (fallback),
    then score with FinBERT (primary) or VADER (fallback).

    Returns
    -------
    dict: score, normalised, signal, positive_news, negative_news, engine
    """
    # Try Google News RSS first
    titles = _fetch_google_news(ticker)
    source = "google_rss"

    # Fallback to yfinance if no results
    if not titles:
        titles = _fetch_yfinance_news(ticker)
        source = "yfinance"

    if not titles:
        return _neutral_result()

    pipe   = _get_finbert()
    engine = f"finbert+{source}" if pipe else f"vader+{source}"

    scores:        list[float] = []
    positive_news: list[str]   = []
    negative_news: list[str]   = []

    for title in titles:
        polarity = _finbert_score(title, pipe) if pipe else _vader_score(title)
        scores.append(polarity)

        if polarity > 0.05:
            positive_news.append(f"{title}  [{polarity:+.2f}]")
        elif polarity < -0.05:
            negative_news.append(f"{title}  [{polarity:+.2f}]")

    if not scores:
        return _neutral_result(engine)

    avg        = sum(scores) / len(scores)
    signal     = "POSITIVE" if avg >= _STRONG_POS else ("NEGATIVE" if avg <= _STRONG_NEG else "NEUTRAL")
    normalised = round((avg + 1) / 2, 4)

    return {
        "score":         round(avg, 4),
        "normalised":    normalised,
        "signal":        signal,
        "positive_news": positive_news,
        "negative_news": negative_news,
        "engine":        engine,
    }