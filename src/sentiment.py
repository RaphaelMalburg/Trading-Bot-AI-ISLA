"""
Sentiment analysis via Google Gemini through OpenRouter.

Scores Bitcoin news headlines from -1.0 (bearish) to +1.0 (bullish).
Acts as a safety filter — trades are blocked when sentiment < SENTIMENT_FLOOR.
"""

import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")


def analyze_sentiment(headlines: list[str], api_key: str = "") -> float:
    """
    Send headlines to Gemini via OpenRouter and return a sentiment score.

    Returns a float in [-1.0, 1.0]. Returns 0.0 (neutral) on any failure.
    """
    if not headlines:
        return 0.0

    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        logger.warning("No OPENROUTER_API_KEY — skipping sentiment")
        return 0.0

    text = "\n".join(f"- {h}" for h in headlines)
    prompt = (
        "Analyze the following Bitcoin/crypto headlines.\n\n"
        f"{text}\n\n"
        "Reply ONLY with a float between -1.0 (extremely bearish) and 1.0 (extremely bullish). "
        "No explanation, just the number."
    )

    try:
        resp = requests.post(
            _OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": _MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
            timeout=15,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            score = max(-1.0, min(1.0, float(content)))
            logger.info("Sentiment score: %.2f", score)
            return score
        logger.error("OpenRouter error %d: %s", resp.status_code, resp.text[:200])
    except (ValueError, KeyError) as e:
        logger.warning("Could not parse sentiment response: %s", e)
    except Exception as e:
        logger.error("Sentiment request failed: %s", e)
    return 0.0
