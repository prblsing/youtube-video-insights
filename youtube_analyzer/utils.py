import re


def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return video_id.group(1) if video_id else None
