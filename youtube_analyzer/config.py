import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# YouTube API settings
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Model settings
SUMMARIZATION_MODEL_FB = os.getenv('SUMMARIZATION_MODEL_FB')
SUMMARIZATION_MODEL_GL = os.getenv('SUMMARIZATION_MODEL_GL')
SENTIMENT_MODEL = os.getenv('SENTIMENT_MODEL')
LLM_MODEL_HG = os.getenv('LLM_MODEL_HG')
LLM_MODEL_FB = os.getenv('LLM_MODEL_FB')
LLM_MODEL_MS = os.getenv('LLM_MODEL_MS')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='youtube_analyzer.log'
)
logger = logging.getLogger(__name__)