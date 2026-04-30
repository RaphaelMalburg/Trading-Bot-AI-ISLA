import os
import logging
import requests
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# Google's LLM used for sentiment analysis via OpenRouter
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")

logger = logging.getLogger(__name__)


def analyze_sentiment(news_headlines):
    """
    Analyze the sentiment of a list of headlines using the Gemini LLM via OpenRouter.

    The headlines are sent to the LLM, which classifies the overall crypto market mood.
    The score acts as a 'safety filter' for the trading bot — trades are vetoed when
    sentiment is too bearish (below SENTIMENT_FLOOR).

    Args:
        news_headlines (list of str): List of headline strings.

    Returns:
        float: A score between -1.0 (extremely bearish) and 1.0 (extremely bullish).
               Returns 0.0 (neutral) on error or when no headlines are provided.
    """
    if not news_headlines:
        return 0.0  # Neutral when no news to analyze

    headlines_text = "\n".join([f"- {h}" for h in news_headlines])
    prompt = f"""
    Analyze the following financial headlines about Bitcoin and cryptocurrencies:

    {headlines_text}

    Classify the overall market sentiment based on them.
    Reply ONLY with a float between -1.0 (extremely bearish) and 1.0 (extremely bullish).
    No explanations — just the number.
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1  # Low temperature for deterministic responses
    }

    try:
        logger.info("Querying Gemini (%s) via OpenRouter...", OPENROUTER_MODEL)
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=15)

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()

            try:
                score = float(content)
                # Clamp to [-1.0, 1.0]
                score = max(-1.0, min(1.0, score))
                logger.info("Sentiment score: %.2f", score)
                return score
            except ValueError:
                logger.warning("Non-numeric Gemini response: %s", content)
                return 0.0
        else:
            logger.error("OpenRouter API error: %s - %s", response.status_code, response.text)
            return 0.0

    except Exception as e:
        logger.error("Sentiment request failed: %s", e)
        return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    bullish_test = [
        "Bitcoin reaches new all-time high above $75,000",
        "Institutional investors increase crypto exposure",
        "SEC approves new spot Bitcoin ETF"
    ]
    print(f"Bullish test: {analyze_sentiment(bullish_test)}")

    bearish_test = [
        "Binance faces regulatory action in three countries",
        "Bitcoin drops 10% after Fed rate-hike speech",
        "Hackers steal $50M from DeFi protocol"
    ]
    print(f"Bearish test: {analyze_sentiment(bearish_test)}")
