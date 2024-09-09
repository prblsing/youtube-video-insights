from transformers import pipeline
from deepmultilingualpunctuation import PunctuationModel
from youtube_analyzer.config import SUMMARIZATION_MODEL_FB
import re
import nltk
from nltk.data import find
from nltk import sent_tokenize


def download_punkt_if_needed():
    try:
        # Check if the 'punkt' tokenizer is already downloaded
        find('tokenizers/punkt.zip')
    except LookupError:
        # Download 'punkt' tokenizer if it's not available
        nltk.download('punkt')


download_punkt_if_needed()

# Load summarization and punctuation restoration models
summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_FB)
punctuator = PunctuationModel()


class ContentAnalysis:
    def __init__(self):
        self.punctuator = punctuator

    def generate_concise_summary(self, text):
        """
        Generate a concise summary with a maximum of 1000-1200 characters.
        Trim the input first, then summarize, and restore punctuation.
        """
        # Trim input to max 3000 characters and to the nearest sentence
        if len(text) > 5000:
            text = self._trim_to_nearest_sentence(text[:5000])

        # Split the text into smaller chunks if it's too long
        chunks = self._split_into_chunks(text, max_tokens=1024)
        summaries = []

        # Summarize each chunk and store the results
        for chunk in chunks:
            summary = summarizer(chunk, max_length=500, min_length=150, do_sample=False)[0]['summary_text']
            summaries.append(summary)

        # Combine all summaries
        combined_summary = " ".join(summaries)

        # Clean the final summary
        cleaned_summary = clean_special_characters(combined_summary)

        # If the cleaned summary exceeds 1200 characters, trim to the nearest sentence
        if len(cleaned_summary) > 3000:
            cleaned_summary = self._trim_to_nearest_sentence(cleaned_summary[:3000])

        # Restore punctuation
        punctuated_summary = self.punctuator.restore_punctuation(cleaned_summary)

        return punctuated_summary

    def format_transcript(self, transcript):
        """
        Formats the transcript using punctuation restoration and adds paragraph breaks.
        """
        chunks = self._split_into_chunks(transcript, max_tokens=1024)
        formatted_transcript = []

        for chunk in chunks:
            punctuated_chunk = self.punctuator.restore_punctuation(chunk)
            formatted_transcript.append(punctuated_chunk)

        return "\n\n".join(formatted_transcript)

    def _split_into_chunks(self, text, max_tokens):
        """
        Splits the text into smaller chunks to ensure they fit within the model's token limit.
        """
        words = text.split()
        chunks, current_chunk = [], []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_tokens:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _trim_to_nearest_sentence(self, text):
        """
        Trims text to the nearest sentence to avoid cutting off mid-sentence.
        """
        sentences = sent_tokenize(text)
        trimmed_text = ""
        for sentence in sentences:
            if len(trimmed_text) + len(sentence) <= len(text):
                trimmed_text += sentence + " "
            else:
                break
        return trimmed_text.strip()


def clean_special_characters(text):
    """
    Removes special characters and extra whitespace.
    """
    return re.sub(r"[^\w\s]", "", text)
