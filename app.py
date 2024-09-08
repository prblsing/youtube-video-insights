import streamlit as st
from youtube_analyzer.youtube_interaction import YouTubeInteraction
from youtube_analyzer.content_analysis import ContentAnalysis
from youtube_analyzer.sentiment_analysis import SentimentAnalysis
from youtube_analyzer.utils import extract_video_id


def main():
    # Apply custom CSS for the theme
    st.markdown("""
        <style>
            body {
                background-color: #2C3E50;
            }
            .stTextInput label {
                color: #F39C12;
            }
            .stTextInput input {
                background-color: #fff;
                color: #2C3E50;
                border-radius: 8px;
                padding: 8px;
                font-size: 16px;
            }
            .stButton button {
                background-color: #F39C12;
                color: white;
                border-radius: 8px;
                font-size: 16px;
            }
            .stButton button:disabled {
                background-color: #D35400;
                color: #AAB7B8;
                cursor: not-allowed;
            }
            .stDownloadButton button {
                background-color: #F39C12;
                color: white;
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("YouTube Video Analyzer")

    youtube_interaction = YouTubeInteraction()
    content_analysis = ContentAnalysis()
    sentiment_analysis = SentimentAnalysis()

    video_url = st.text_input("Enter YouTube Video URL:")
    video_id = extract_video_id(video_url) if video_url else None

    # Disable the refresh button while no video URL is entered or analysis is ongoing
    if video_url and video_id:
        refresh_button_disabled = False
    else:
        refresh_button_disabled = True

    if st.button("Refresh", disabled=refresh_button_disabled):
        st.experimental_rerun()

    if video_url:
        if video_id:
            with st.spinner("Analyzing video..."):
                transcript = youtube_interaction.get_transcript(video_id)
                if transcript:
                    comments = youtube_interaction.get_comments(video_id)

                    # Generate summary
                    summary = content_analysis.generate_concise_summary(transcript)

                    # Full transcript
                    formatted_transcript = content_analysis.format_transcript(transcript)

                    # Sentiment analysis
                    sentiment_result = sentiment_analysis.analyze_sentiment(comments)
                    sentiment_score = sentiment_result["average_sentiment_score"]
                    effectiveness = sentiment_result["effectiveness"]
                    most_positive_comment = sentiment_result["most_positive_comment"]
                    most_engaging_comment = sentiment_result["most_engaging_comment"]

                    # Display the results in different tabs
                    tab1, tab2, tab3 = st.tabs(["Brief Summary", "Full Transcript", "Sentiment Analysis"])

                    # Tab for the Brief Summary
                    with tab1:
                        st.header("Brief Summary")
                        st.write(summary)

                    # Tab for Full Transcript
                    with tab2:
                        st.header("Full Transcript")
                        st.text_area("Transcript", formatted_transcript, height=400)
                        # Add a button to download the transcript as a text file
                        st.download_button(label="Download Transcript", data=formatted_transcript,
                                           file_name="transcript.txt", mime="text/plain")

                    # Tab for Sentiment Analysis
                    with tab3:
                        st.header("Sentiment Analysis and Effectiveness")
                        st.write(f"Average Sentiment Score: {sentiment_score:.2f}")
                        st.write(f"Video Effectiveness: {effectiveness}")
                        st.subheader("Most Positive Comment")
                        st.markdown(most_positive_comment, unsafe_allow_html=True)
                        st.subheader("Most Engaging Comment")
                        st.markdown(most_engaging_comment, unsafe_allow_html=True)
                else:
                    st.error("Unable to fetch video transcript. Please check the video URL and try again.")
        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")


if __name__ == "__main__":
    main()
