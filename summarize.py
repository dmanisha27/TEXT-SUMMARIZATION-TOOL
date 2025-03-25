import nltk
import numpy as np
import re
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}\n")


def summarize_nltk(text, num_sentences=3):
    """Summarizes text using NLTK word frequency method."""
    print("Running NLTK Frequency-Based Summarization...")

    sentences = sent_tokenize(text)
    if not sentences:
        print("No sentences detected. Please check input text.")
        return "Unable to generate summary."

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    word_frequencies = {}
    for word in words:
        if word.isalnum() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    max_freq = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_freq  # Normalize frequencies

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    summary = " ".join(ranked_sentences[:min(num_sentences, len(ranked_sentences))])
    return summary if summary else "No summary generated."


def summarize_tfidf(text, num_sentences=3):
    """Summarizes text using TF-IDF scoring."""
    print("Running TF-IDF Summarization...")

    sentences = sent_tokenize(text)
    if not sentences:
        print("No sentences detected. Please check input text.")
        return "Unable to generate summary."

    vectorizer = TfidfVectorizer(stop_words="english")
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError as e:
        print(f"TF-IDF Error: {e}")
        return "Unable to process text."

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    if len(sentence_scores) != len(sentences):
        print(f"Mismatch: {len(sentence_scores)} scores vs {len(sentences)} sentences.")
        return "Processing error. Try different text."

    ranked_sentences = sorted(zip(sentence_scores, sentences), key=lambda x: x[0], reverse=True)

    num_sentences = min(num_sentences, len(ranked_sentences))
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    return " ".join(summary_sentences) if summary_sentences else "No summary generated."


def summarize_transformer(text):
    """Summarizes text using a deep-learning Transformer model from Hugging Face."""
    print("Running Transformer-Based Summarization...")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
    
    try:
        summary = summarizer(text, max_length=250, min_length=50, do_sample=False)
    except Exception as e:
        print(f"Transformer Model Error: {e}")
        return "Unable to generate summary."

    return summary[0]['summary_text'] if summary else "No summary generated."


if __name__ == "__main__":
    print("\nAI Text Summarization Tool")
    print("Choose an input method:")
    print("1. Enter text manually")
    print("2. Read from a file (sample.txt)")
    
    choice = input("\nEnter choice (1/2): ").strip()

    text = ""

    if choice == "1":
        text = input("\nEnter the text to summarize:\n").strip()
    elif choice == "2":
        print("Loading text from sample.txt...")
        try:
            with open("sample.txt", "r", encoding="utf-8") as file:
                text = file.read().strip()
            print(f"Loaded {len(text.split())} words from sample.txt.")
        except FileNotFoundError:
            print("Error: sample.txt not found.")
            exit()
    else:
        print("Invalid choice. Exiting...")
        exit()

    if not text:
        print("Error: No text provided.")
        exit()

    print("\nChoose summarization method:")
    print("1. NLTK Frequency-Based")
    print("2. TF-IDF-Based")
    print("3. Transformer-Based (Deep Learning)")

    method = input("\nEnter choice (1/2/3): ").strip()

    print("\nGenerating summary...\n")

    summary = ""
    
    if method == "1":
        summary = summarize_nltk(text)
    elif method == "2":
        summary = summarize_tfidf(text)
    elif method == "3":
        summary = summarize_transformer(text)
    else:
        print("Invalid choice. Exiting...")
        exit()

    print("\nSummary Output\n")
    print(summary)
