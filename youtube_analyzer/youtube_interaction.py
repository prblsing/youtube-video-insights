from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from youtube_analyzer.config import YOUTUBE_API_KEY, logger


class YouTubeInteraction:
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    def get_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry['text'] for entry in transcript])
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
            return None

    def get_comments(self, video_id, max_results=250):
        try:
            comments = []
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results
            )
            while request and len(comments) < max_results:
                response = request.execute()
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                request = self.youtube.commentThreads().list_next(request, response)
            return comments[:max_results]
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")
            return []
