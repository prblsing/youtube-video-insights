import os
from dotenv import load_dotenv
import logging

# Load environment variables from the .env file if available
load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', os.environ.get('YOUTUBE_API_KEY'))

# Model settings
SUMMARIZATION_MODEL_FB = os.getenv('SUMMARIZATION_MODEL_FB', os.environ.get('SUMMARIZATION_MODEL_FB'))
# SUMMARIZATION_MODEL_GL = os.getenv('SUMMARIZATION_MODEL_GL', os.environ.get('SUMMARIZATION_MODEL_GL'))
SENTIMENT_MODEL = os.getenv('SENTIMENT_MODEL', os.environ.get('SENTIMENT_MODEL'))
# LLM_MODEL_HG = os.getenv('LLM_MODEL_HG', os.environ.get('LLM_MODEL_HG'))
# LLM_MODEL_FB = os.getenv('LLM_MODEL_FB', os.environ.get('LLM_MODEL_FB'))

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='youtube_analyzer.log'
)
logger = logging.getLogger(__name__)

# To confirm the environment variables are being read correctly
logger.info(f"YOUTUBE_API_KEY: {len(YOUTUBE_API_KEY)}")
logger.info(f"SUMMARIZATION_MODEL_FB: {len(SUMMARIZATION_MODEL_FB)}")
logger.info(f"SENTIMENT_MODEL: {len(SENTIMENT_MODEL)}")
