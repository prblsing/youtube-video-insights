from transformers import pipeline, AutoTokenizer
from deepmultilingualpunctuation import PunctuationModel
from youtube_analyzer.config import SUMMARIZATION_MODEL_FB
import re
import nltk
from nltk.data import find
from nltk import sent_tokenize

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_punkt_if_needed():
    try:
        # Check if the 'punkt' tokenizer is already downloaded
        find('tokenizers/punkt.zip')
    except LookupError:
        # Download 'punkt' tokenizer if it's not available
        nltk.download('punkt')

download_punkt_if_needed()

try:
    # Load summarization model and tokenizer
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_FB)
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_FB)
except Exception as e:
    logger.error(f"Error loading summarization model or tokenizer: {str(e)}")
    summarizer = None
    tokenizer = None

try:
    # Load punctuation restoration model
    punctuator = PunctuationModel()
except Exception as e:
    logger.error(f"Error loading punctuation model: {str(e)}")
    punctuator = None

class ContentAnalysis:
    def __init__(self):
        self.punctuator = punctuator
        self.summarizer = summarizer
        self.tokenizer = tokenizer
        self.model_max_length = self.tokenizer.model_max_length if self.tokenizer else 1024
        # To leave room for summary tokens, set chunk size smaller than max_length
        self.chunk_size = 512  # Adjust based on model's max_length and summary length

    def generate_concise_summary(self, text):
        """
        Generate a concise summary with a maximum of 1000-1200 characters.
        If an error occurs, return a user-friendly message.
        """
        if not self.summarizer or not self.tokenizer or not self.punctuator:
            logger.error("Models are not properly loaded.")
            return "Internal error: Unable to process the request."

        try:
            # Optionally, restore punctuation before processing
            # text = self.punctuator.restore_punctuation(text)

            # Trim input to max 5000 characters and to the nearest sentence
            if len(text) > 5000:
                text = self._trim_to_nearest_sentence(text[:5000])
                logger.info("Input text trimmed to 5000 characters.")

            # Split the text into smaller chunks based on token counts
            chunks = self._split_into_token_chunks(text, max_tokens=self.chunk_size)
            logger.info(f"Text split into {len(chunks)} chunks.")

            summaries = []

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    logger.warning(f"Chunk {idx} is empty. Skipping.")
                    continue  # skip empty chunks
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=200,  # Adjust based on desired summary length
                        min_length=50,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)
                    logger.info(f"Chunk {idx} summarized successfully.")
                except IndexError as e:
                    logger.error(f"IndexError summarizing chunk {idx}: {str(e)}")
                    return "Unable to generate summary for this video. Please try another video."
                except Exception as e:
                    logger.error(f"Error summarizing chunk {idx}: {str(e)}")
                    continue

            if not summaries:
                logger.warning("No summaries were generated from the chunks.")
                return "Unable to generate summary for this video. Please try another video."

            # Combine all summaries
            combined_summary = " ".join(summaries)
            logger.info("Combined summaries from all chunks.")

            # Clean the final summary
            cleaned_summary = clean_special_characters(combined_summary)
            logger.info("Cleaned combined summary.")

            # Trim the summary to 1200 characters, to the nearest sentence
            if len(cleaned_summary) > 1200:
                cleaned_summary = self._trim_to_nearest_sentence(cleaned_summary[:1200])
                logger.info("Combined summary trimmed to 1200 characters.")

            # Restore punctuation
            punctuated_summary = self.punctuator.restore_punctuation(cleaned_summary)
            logger.info("Punctuation restored in summary.")

            return punctuated_summary

        except IndexError as e:
            logger.error(f"IndexError generating summary: {str(e)}")
            return "Unable to generate summary for this video. Please try another video."
        except Exception as e:
            # Log the error
            logger.error(f"Error generating summary: {str(e)}")
            # Return a user-friendly message
            return "Unable to generate summary for this video. Please try another video."

    def format_transcript(self, transcript):
        """
        Formats the transcript using punctuation restoration and adds paragraph breaks.
        """
        if not self.punctuator:
            logger.error("Punctuation model is not loaded.")
            return "Internal error: Unable to format transcript."

        try:
            chunks = self._split_into_token_chunks(transcript, max_tokens=self.chunk_size)
            logger.info(f"Transcript split into {len(chunks)} chunks for formatting.")
            formatted_transcript = []

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    logger.warning(f"Transcript chunk {idx} is empty. Skipping.")
                    continue
                try:
                    punctuated_chunk = self.punctuator.restore_punctuation(chunk)
                    formatted_transcript.append(punctuated_chunk)
                    logger.info(f"Transcript chunk {idx} punctuated successfully.")
                except Exception as e:
                    logger.error(f"Error formatting transcript chunk {idx}: {str(e)}")
                    continue

            return "\n\n".join(formatted_transcript)

        except Exception as e:
            logger.error(f"Error formatting transcript: {str(e)}")
            return "Unable to format transcript."

    def _split_into_token_chunks(self, text, max_tokens):
        """
        Splits the text into chunks based on token counts using the tokenizer.
        """
        if not self.tokenizer:
            logger.error("Tokenizer is not loaded.")
            return []

        try:
            tokens = self.tokenizer.encode(text, truncation=False)
            logger.info(f"Total tokens in text: {len(tokens)}")
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                chunks.append(chunk_text)
            return chunks
        except Exception as e:
            logger.error(f"Error during tokenization and chunking: {str(e)}")
            return []

    def _trim_to_nearest_sentence(self, text):
        """
        Trims text to the nearest sentence to avoid cutting off mid-sentence.
        """
        try:
            sentences = sent_tokenize(text)
            trimmed_text = ""
            for sentence in sentences:
                if len(trimmed_text) + len(sentence) + 1 <= len(text):
                    trimmed_text += sentence + " "
                else:
                    break
            return trimmed_text.strip()
        except Exception as e:
            logger.error(f"Error trimming text to nearest sentence: {str(e)}")
            return text  # Fallback to original text if trimming fails

def clean_special_characters(text):
    """
    Removes special characters and extra whitespace.
    """
    try:
        cleaned_text = re.sub(r"[^\w\s]", "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"Error cleaning special characters: {str(e)}")
        return text  # Fallback to original text if cleaning fails
