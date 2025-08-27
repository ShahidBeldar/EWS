from transformers import pipeline
from utils import load_data, compute_similarity, compute_sentiment, sentiment_model

def analyze_headline(headline):
    """
    Analyze sentiment of a single financial news headline.
    Returns polarity, impact message, and top 3 similar headlines.
    """
    try:
        # Run sentiment analysis
        result = sentiment_model(headline, truncation=True, max_length=128)[0]
        label = result['label']
        score = result['score']

        # Convert to polarity & impact message (adjust for TinyBERT labels)
        if label in ['LABEL_1', 'POSITIVE']:  # Positive
            impact = f"Positive ({score:.2f} confidence) - Price likely to go up"
            polarity = score
        elif label in ['LABEL_0', 'NEGATIVE']:  # Negative
            impact = f"Negative ({score:.2f} confidence) - Price likely to go down"
            polarity = -score
        else:  # LABEL_2 or neutral
            impact = f"Neutral ({score:.2f} confidence) - Price impact unclear"
            polarity = 0.0

        # Load dataset and compute similarity with historical headlines
        news_df = load_data("news.csv")
        news_df = compute_sentiment(news_df)  # Apply transformer-based sentiment
        matched = compute_similarity(news_df, headline)

        return {
            'polarity': polarity,
            'impact': impact,
            'matched': matched[['Date', 'Headline', 'sentiment', 'similarity']].head(3)
        }
    except Exception as e:
        return {
            'polarity': 0.0,
            'impact': f"Error analyzing headline: {str(e)}",
            'matched': None
        }