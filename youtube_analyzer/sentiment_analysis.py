from transformers import pipeline
from youtube_analyzer.config import SENTIMENT_MODEL, logger


class SentimentAnalysis:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    def analyze_sentiment(self, texts):
        try:
            results = self.sentiment_analyzer(texts)
            scores = [result['score'] if result['label'] == 'POSITIVE' else 1 - result['score'] for result in results]
            return sum(scores) / len(scores) if scores else 0
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0

    def evaluate_effectiveness(self, sentiment_score):
        if sentiment_score > 0.7:
            return "Very Effective"
        elif sentiment_score > 0.5:
            return "Effective"
        elif sentiment_score > 0.3:
            return "Somewhat Effective"
        else:
            return "Not Effective"

    def get_most_positive_comment(self, comments):
        scored_comments = [(comment, self.sentiment_analyzer(comment)[0]['score']) for comment in comments]
        return max(scored_comments, key=lambda x: x[1])[0]

    def get_most_engaging_comment(self, comments):
        return max(comments, key=len)
