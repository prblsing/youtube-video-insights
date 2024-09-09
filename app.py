import streamlit as st
import sys  # Required for monkey-patching
from youtube_analyzer.youtube_interaction import YouTubeInteraction
from youtube_analyzer.content_analysis import ContentAnalysis
from youtube_analyzer.sentiment_analysis import SentimentAnalysis
from youtube_analyzer.utils import extract_video_id
import logging

logger = logging.getLogger(__name__)

# Custom Batman-themed exception handler
def exception_handler(e):
    """
    Custom error handling for uncaught exceptions.
    Displays an image of a masked character and a friendly error message when things go wrong.
    """
    # Display Batman-themed image when an error occurs
    st.image(
        'https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDY0ZHFkMHA5N21hanpnZGpnazNsNGg4ZWJ1ZnptMmdqaDJiNDVwbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6la5HPkiGrzFlEUYHp/giphy.gif',
        caption="I need a sidekick!"
    )

    # Friendly message to indicate an issue, like Batman calling for a sidekick to solve the problem
    st.error(
        f"Oops! Something went wrong with a {type(e).__name__}. Looks like even Batman needs a sidekick sometimes to fix things! 🦇"
    )

    # Write the full error to help with debugging in case needed (for development environments)
    st.write(f"Full error details: {e}")

    # Re-raise the error to ensure it's logged appropriately in console or monitoring tools
    raise e

# Monkey-patch streamlit's error handling to use the custom Batman-themed handler
error_util = sys.modules['streamlit.error_util']
error_util.handle_uncaught_app_exception.__code__ = exception_handler.__code__

# Application code continues
def main():
    st.title("YouTube Video Analyzer")

    youtube_interaction = YouTubeInteraction()
    content_analysis = ContentAnalysis()
    sentiment_analysis = SentimentAnalysis()

    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        video_id = extract_video_id(video_url)

        if video_id:
            with st.spinner("Analyzing video..."):
                try:
                    # Fetch transcript
                    transcript = youtube_interaction.get_transcript(video_id)

                    if transcript:
                        try:
                            # Generate summary
                            summary = content_analysis.generate_concise_summary(transcript)
                        except Exception as e:
                            st.error("Error generating summary. Please try another video.")
                            logger.error(f"Error generating summary: {str(e)}")
                            return

                        # Full transcript
                        try:
                            formatted_transcript = content_analysis.format_transcript(transcript)
                        except Exception as e:
                            st.error("Error formatting transcript. Please try again later.")
                            logger.error(f"Error formatting transcript: {str(e)}")
                            return

                        # Fetch comments
                        comments = youtube_interaction.get_comments(video_id)

                        if comments:
                            # Sentiment analysis
                            try:
                                sentiment_result = sentiment_analysis.analyze_sentiment(comments)
                                sentiment_score = sentiment_result["average_sentiment_score"]
                                effectiveness = sentiment_result["effectiveness"]
                                most_positive_comment = sentiment_result["most_positive_comment"]
                                most_engaging_comment = sentiment_result["most_engaging_comment"]
                            except Exception as e:
                                st.error("Error analyzing sentiment. Please try again later.")
                                logger.error(f"Error analyzing sentiment: {str(e)}")
                                return

                            # Tabs for results
                            tab1, tab2, tab3 = st.tabs(["Brief Summary", "Full Transcript", "Sentiment Analysis"])

                            with tab1:
                                st.header("Brief Summary")
                                st.write(summary)

                            with tab2:
                                st.header("Full Transcript")
                                st.text_area("Transcript", formatted_transcript, height=400)
                                st.download_button(label="Download Transcript", data=formatted_transcript,
                                                   file_name="transcript.txt", mime="text/plain")

                            with tab3:
                                st.header("Sentiment Analysis and Effectiveness")
                                st.write(f"Average Sentiment Score: {sentiment_score:.2f}")
                                st.write(f"Video Effectiveness: {effectiveness}")
                                st.subheader("Most Positive Comment")
                                st.markdown(most_positive_comment, unsafe_allow_html=True)
                                st.subheader("Most Engaging Comment")
                                st.markdown(most_engaging_comment, unsafe_allow_html=True)
                        else:
                            st.warning("No comments found for this video.")

                    else:
                        st.error("Unable to fetch video transcript. Please check the video URL and try again.")

                except Exception as e:
                    # This will call the custom Batman-themed error handler
                    st.error("An unexpected error occurred during video analysis. Please try again later.")
                    logger.error(f"Error during video analysis: {str(e)}")

        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")

if __name__ == "__main__":
    main()
