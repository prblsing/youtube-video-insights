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
        find('tokenizers/punkt.zip')
    except LookupError:
        nltk.download('punkt')

download_punkt_if_needed()

class ContentAnalysis:
    def __init__(self):
        try:
            self.summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_FB)
            self.tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_FB)
            self.punctuator = PunctuationModel()
            self.model_max_length = self.tokenizer.model_max_length
            self.chunk_size = min(512, self.model_max_length // 2)  # Adjust based on model's max_length
        except Exception as e:
            logger.error(f"Error initializing ContentAnalysis: {str(e)}")
            self.summarizer = None
            self.tokenizer = None
            self.punctuator = None

    def generate_concise_summary(self, text):
        if not self.summarizer or not self.tokenizer or not self.punctuator:
            return "Unable to generate summary due to initialization error. Please try again later."

        try:
            text = self._preprocess_text(text)
            chunks = self._split_into_token_chunks(text)
            summaries = self._summarize_chunks(chunks)
            
            if not summaries:
                return "Unable to generate summary for this video. Please try another video."

            combined_summary = " ".join(summaries)
            cleaned_summary = self._clean_and_trim_summary(combined_summary)
            punctuated_summary = self.punctuator.restore_punctuation(cleaned_summary)

            return punctuated_summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary for this video. Please try another video."

    def _preprocess_text(self, text):
        if len(text) > 5000:
            text = self._trim_to_nearest_sentence(text[:5000])
        return text

    def _split_into_token_chunks(self, text):
        try:
            tokens = self.tokenizer.encode(text, truncation=False)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            return chunks
        except Exception as e:
            logger.error(f"Error during tokenization and chunking: {str(e)}")
            return [text]  # Return the original text as a single chunk if tokenization fails

    def _summarize_chunks(self, chunks):
        summaries = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                summary = self.summarizer(
                    chunk,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing chunk {idx}: {str(e)}")
        return summaries

    def _clean_and_trim_summary(self, summary):
        cleaned_summary = re.sub(r"[^\w\s]", "", summary)
        cleaned_summary = re.sub(r"\s+", " ", cleaned_summary).strip()
        if len(cleaned_summary) > 1200:
            cleaned_summary = self._trim_to_nearest_sentence(cleaned_summary[:1200])
        return cleaned_summary

    def _trim_to_nearest_sentence(self, text):
        sentences = sent_tokenize(text)
        trimmed_text = ""
        for sentence in sentences:
            if len(trimmed_text) + len(sentence) + 1 <= len(text):
                trimmed_text += sentence + " "
            else:
                break
        return trimmed_text.strip()

    def format_transcript(self, transcript):
        if not self.punctuator:
            return "Unable to format transcript due to initialization error."

        try:
            chunks = self._split_into_token_chunks(transcript)
            formatted_chunks = [self.punctuator.restore_punctuation(chunk) for chunk in chunks if chunk.strip()]
            return "\n\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error formatting transcript: {str(e)}")
            return "Unable to format transcript."
