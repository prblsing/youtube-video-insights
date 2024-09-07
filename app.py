import streamlit as st
from youtube_analyzer.youtube_interaction import YouTubeInteraction
from youtube_analyzer.content_analysis import ContentAnalysis
from youtube_analyzer.sentiment_analysis import SentimentAnalysis
from youtube_analyzer.utils import extract_video_id


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
                transcript = youtube_interaction.get_transcript(video_id)
                if transcript:
                    comments = youtube_interaction.get_comments(video_id)

                    # Generate summary
                    summary = content_analysis.generate_concise_summary(transcript)

                    # Full transcript
                    formatted_transcript = content_analysis.format_transcript(transcript)

                    # Sentiment analysis
                    sentiment_score = sentiment_analysis.analyze_sentiment(comments)
                    effectiveness = sentiment_analysis.evaluate_effectiveness(sentiment_score)
                    most_positive_comment = sentiment_analysis.get_most_positive_comment(comments)
                    most_engaging_comment = sentiment_analysis.get_most_engaging_comment(comments)

                    # Clean up comment text to remove special characters
                    most_engaging_comment = most_engaging_comment.replace("<br>", "")
                    most_positive_comment = most_positive_comment.replace("<br>", "")

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
                        st.write(most_positive_comment)
                        st.subheader("Most Engaging Comment")
                        st.write(most_engaging_comment)
                else:
                    st.error("Unable to fetch video transcript. Please check the video URL and try again.")
        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")


if __name__ == "__main__":
    main()
