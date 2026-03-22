from textblob import TextBlob
import yfinance as yf


def get_news_sentiment(ticker):

    stock = yf.Ticker(ticker)

    try:
        news = stock.get_news()
    except:
        news = []

    if not news:
        return {
            "score": 0,
            "signal": "NEUTRAL",
            "positive_news": [],
            "negative_news": []
        }

    scores = []
    positive_news = []
    negative_news = []

    for article in news[:10]:
        title = article.get("title", "")
        
        polarity = TextBlob(title).sentiment.polarity

        scores.append(polarity)

        if polarity > 0:
            positive_news.append(title)
        elif polarity < 0:
            negative_news.append(title)

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.3:
        signal = "POSITIVE"
    elif avg_score < -0.3:
        signal = "NEGATIVE"
    else:
        signal = "NEUTRAL"

    return {
        "score": round(avg_score, 2),
        "signal": signal,
        "positive_news": positive_news,
        "negative_news": negative_news
    }