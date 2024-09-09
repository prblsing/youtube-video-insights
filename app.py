import streamlit as st
from youtube_analyzer.youtube_interaction import YouTubeInteraction
from youtube_analyzer.content_analysis import ContentAnalysis
from youtube_analyzer.sentiment_analysis import SentimentAnalysis
from youtube_analyzer.utils import extract_video_id
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("YouTube Video Analyzer")

    # Initialize components
    youtube_interaction = YouTubeInteraction()
    content_analysis = ContentAnalysis()
    sentiment_analysis = SentimentAnalysis()

    # User input
    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            with st.spinner("Analyzing video..."):
                try:
                    # Fetch transcript
                    transcript = youtube_interaction.get_transcript(video_id)
                    if transcript:
                        # Fetch comments
                        comments = youtube_interaction.get_comments(video_id)

                        # Generate summary
                        summary = content_analysis.generate_concise_summary(transcript)

                        # Format full transcript
                        formatted_transcript = content_analysis.format_transcript(transcript)

                        # Perform sentiment analysis
                        sentiment_result = sentiment_analysis.analyze_sentiment(comments)

                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["Brief Summary", "Full Transcript", "Sentiment Analysis"])

                        with tab1:
                            st.header("Brief Summary")
                            if "Unable to generate summary" in summary:
                                st.warning(summary)
                            else:
                                st.write(summary)

                        with tab2:
                            st.header("Full Transcript")
                            if "Unable to format transcript" in formatted_transcript:
                                st.warning(formatted_transcript)
                            else:
                                st.text_area("Transcript", formatted_transcript, height=400)
                                st.download_button(
                                    label="Download Transcript",
                                    data=formatted_transcript,
                                    file_name="transcript.txt",
                                    mime="text/plain"
                                )

                        with tab3:
                            st.header("Sentiment Analysis and Effectiveness")
                            if sentiment_result:
                                st.write(f"Average Sentiment Score: {sentiment_result['average_sentiment_score']:.2f}")
                                st.write(f"Video Effectiveness: {sentiment_result['effectiveness']}")
                                
                                st.subheader("Most Positive Comment")
                                if sentiment_result['most_positive_comment']:
                                    st.markdown(sentiment_result['most_positive_comment'], unsafe_allow_html=True)
                                else:
                                    st.info("No positive comments found.")

                                st.subheader("Most Engaging Comment")
                                if sentiment_result['most_engaging_comment']:
                                    st.markdown(sentiment_result['most_engaging_comment'], unsafe_allow_html=True)
                                else:
                                    st.info("No engaging comments found.")
                            else:
                                st.warning("Unable to perform sentiment analysis. This may be due to a lack of comments or an error in processing.")

                    else:
                        st.error("Unable to fetch video transcript. Please check if the video has captions available and try again.")

                except Exception as e:
                    st.error("An unexpected error occurred during video analysis. Please try again later.")
                    logger.error(f"Error during video analysis: {str(e)}")
        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")

if __name__ == "__main__":
    main()
