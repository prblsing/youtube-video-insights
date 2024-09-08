from transformers import pipeline
from youtube_analyzer.config import SENTIMENT_MODEL, logger

class SentimentAnalysis:
    def __init__(self):
        # Load the sentiment analysis model
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    def analyze_sentiment(self, comments):
        total_score = 0
        sentiment_count = 0
        most_positive_comment = None
        most_engaging_comment = None
        highest_sentiment_score = 0
        max_engagement = 0

        for comment in comments:
            # Truncate comment if needed
            truncated_comment = self._truncate_text(comment)
            try:
                # Perform sentiment analysis on the truncated comment
                sentiment_result = self.sentiment_analyzer(truncated_comment)[0]

                # Calculate sentiment score
                score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else 1 - sentiment_result['score']
                total_score += score
                sentiment_count += 1

                # Track the most positive comment
                if score > highest_sentiment_score:
                    highest_sentiment_score = score
                    most_positive_comment = comment

                # Track the most engaging comment (based on length or engagement metric)
                if len(comment) > max_engagement:
                    max_engagement = len(comment)
                    most_engaging_comment = comment

            except Exception as e:
                logger.error(f"Error analyzing sentiment for comment: {e}")

        # Calculate the average sentiment score
        average_sentiment_score = total_score / sentiment_count if sentiment_count > 0 else 0

        # Calculate effectiveness based on the average score
        effectiveness = self.evaluate_effectiveness(average_sentiment_score)

        return {
            "average_sentiment_score": average_sentiment_score,
            "effectiveness": effectiveness,
            "most_positive_comment": most_positive_comment,
            "most_engaging_comment": most_engaging_comment
        }

    def evaluate_effectiveness(self, sentiment_score):
        # Return effectiveness based on sentiment score
        if sentiment_score > 0.7:
            return "Very Effective"
        elif sentiment_score > 0.5:
            return "Effective"
        elif sentiment_score > 0.3:
            return "Somewhat Effective"
        else:
            return "Not Effective"

    def _truncate_text(self, text, max_tokens=512):
        """
        Truncate the text to fit within the token limit (512 tokens).
        """
        return text[:max_tokens]
