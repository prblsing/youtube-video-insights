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
                        # Fetch comments
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

                        # Tabs for results
                        tab1, tab2, tab3 = st.tabs(["Brief Summary", "Full Transcript", "Sentiment Analysis"])

                        with tab1:
                            st.header("Brief Summary")
                            st.write(summary)

                        with tab2:
                            st.header("Full Transcript")
                            st.text_area("Transcript", formatted_transcript, height=400)
                            st.download_button(label="Download Transcript", data=formatted_transcript, file_name="transcript.txt", mime="text/plain")

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
                except Exception as e:
                    st.error("An unexpected error occurred during video analysis. Please try again later.")
                    logger.error(f"Error during video analysis: {str(e)}")
        else:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
