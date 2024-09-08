from transformers import pipeline
from deepmultilingualpunctuation import PunctuationModel
import nltk
from youtube_analyzer.config import SUMMARIZATION_MODEL_FB
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration
import re

model = BartForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL_FB)
tokenizer = BartTokenizer.from_pretrained(SUMMARIZATION_MODEL_FB)


def clean_special_characters(text):
    # Remove special characters and extra whitespace
    return re.sub(r"[^\w\s]", "", text)


def summarize(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0])


# def rephrase_summary(text):
#     pipe = pipeline("text-generation", model=LLM_MODEL_FB)
#     prompt = f"Summarize the following text into a brief overview that explains what this video will cover, " \
#              f"the main topics discussed, and what the viewer can expect to learn: {text} "
#     generated_text = pipe(prompt, max_length=150, num_return_sequences=1, do_sample=False)
#     print(f"{generated_text=}")
#
#     return generated_text[0]["generated_text"]


class ContentAnalysis:
    def __init__(self):
        self.summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_FB)
        # self.summarizer = Summarizer()
        self.punctuator = PunctuationModel()

    def generate_concise_summary(self, text, max_length=100, min_length=30):
        """
        Dynamically generates a concise, high-level overview of the video based on its transcript.
        The overview should capture the main points and provide an understanding of what the video covers.
        """
        try:
            max_tokens = 1024
            chunks = self._split_into_chunks(text, max_tokens)
            summaries = []

            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            # Combine the generated summaries
            # final_summary = " ".join(summaries)
            final_summary = summarize(" ".join(summaries))
            cleaned_summary = clean_special_characters(final_summary.replace("</s>", "").replace("<s>", ""))
            # ret_summ = rephrase_summary(cleaned_summary)
            # print(f"{ret_summ=}")
            print(f"{cleaned_summary=}")

            return cleaned_summary

        except Exception as e:
            print(f"Error generating summary: {e}")
            sentences = sent_tokenize(text)
            # Fallback summary based on the first few sentences in case of an error
            return " ".join(sentences[:3])

    def format_transcript(self, transcript):
        """
        Formats the transcript using an ML-based punctuation restoration model and
        adds paragraph breaks for better readability.
        """
        chunks = self._split_into_chunks(transcript, max_tokens=1024)
        formatted_transcript = []

        for chunk in chunks:
            punctuated_chunk = self.punctuator.restore_punctuation(chunk)
            formatted_transcript.append(punctuated_chunk)

        structured_transcript = "\n\n".join(formatted_transcript)

        return structured_transcript

    def _split_into_chunks(self, text, max_tokens):
        """
        Splits a given text into smaller chunks based on token limit for the punctuator and summarizer.
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_tokens:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
