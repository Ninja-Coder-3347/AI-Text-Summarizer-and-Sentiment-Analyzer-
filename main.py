# Advanced Text Summarizer + Sentiment Analyzer with Attractive GUI and WordCloud Popup
# Author: Pallavi Suryawanshi

import tkinter as tk
from tkinter import messagebox, scrolledtext
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
from wordcloud import WordCloud
from PIL import ImageTk, Image
import io

# ---------------- Text Cleaning ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# ---------------- Text Summarization ----------------
def summarize_text(text, n_sentences=5):
    sentences = text.split('. ')
    clean_sentences = [clean_text(s) for s in sentences]

    if len(sentences) <= n_sentences:
        return text

    vectorizer = CountVectorizer().fit_transform(clean_sentences)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    scores = sim_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[-n_sentences:]]
    return '. '.join(ranked_sentences)

# ---------------- Sentiment Analysis ----------------
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive 😄"
    elif polarity < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# ---------------- WordCloud Popup ----------------
positive_words = ['love','amazing','great','fantastic','masterpiece','wonderful','stellar','brilliant','best','awesome']
negative_words = ['boring','worst','terrible','disappointing','predictable','awful','poor','bad','horrible','flaws']

def show_wordcloud_window(text):
    word_freq = {}
    for word in text.split():
        clean_word = word.lower().strip(string.punctuation)
        if clean_word == "":
            continue
        word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

    def color_func(word, *args, **kwargs):
        if word in positive_words:
            return "#28a745"  # Green
        elif word in negative_words:
            return "#dc3545"  # Red
        else:
            return "#007bff"  # Blue

    wc = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(word_freq)
    image = wc.recolor(color_func=color_func)

    img_buffer = io.BytesIO()
    image.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    tk_image = ImageTk.PhotoImage(Image.open(img_buffer))

    wc_window = tk.Toplevel()
    wc_window.title("🌟 WordCloud")
    wc_window.geometry("820x650")
    wc_window.config(bg="#f0f2f5")

    label = tk.Label(wc_window, image=tk_image, bg="#f0f2f5")
    label.image = tk_image  # Keep reference
    label.pack(pady=10, padx=10)

# ---------------- GUI Function ----------------
def process_text():
    user_text = text_box.get("1.0", tk.END).strip()
    if user_text == "":
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    # Summarize
    summary = summarize_text(user_text, n_sentences=5)
    summary_box.config(state='normal')
    summary_box.delete("1.0", tk.END)
    summary_box.insert(tk.END, summary)
    summary_box.config(state='disabled')

    # Sentiment
    sentiment = analyze_sentiment(user_text)
    sentiment_label.config(text=f"Sentiment: {sentiment}")

    # WordCloud Popup
    show_wordcloud_window(user_text)

# ---------------- Tkinter GUI ----------------
window = tk.Tk()
window.title("📝 Text Summarizer & Sentiment Analyzer")
window.geometry("900x750")
window.config(bg="#f0f2f5")

# Title
title_label = tk.Label(window, text="Text Summarizer + Sentiment + WordCloud",
                       font=("Helvetica", 20, "bold"), bg="#f0f2f5", fg="#4b0082")
title_label.pack(pady=15)

# Input Frame
input_frame = tk.Frame(window, bg="#ffffff", bd=2, relief="groove")
input_frame.pack(pady=10, padx=20, fill="both")

input_label = tk.Label(input_frame, text="Enter your text below:", font=("Arial", 13, "bold"), bg="#ffffff")
input_label.pack(pady=5)

text_box = scrolledtext.ScrolledText(input_frame, height=8, width=95, font=("Arial", 12))
text_box.pack(pady=5)

analyze_button = tk.Button(input_frame, text="Analyze Text", command=process_text,
                           font=("Arial", 13, "bold"), bg="#6f42c1", fg="white", padx=10, pady=5)
analyze_button.pack(pady=10)

# Sentiment Frame
sentiment_frame = tk.Frame(window, bg="#d1e7dd", bd=2, relief="ridge")
sentiment_frame.pack(pady=10, padx=20, fill="x")
sentiment_label = tk.Label(sentiment_frame, text="Sentiment: ", font=("Arial", 14, "bold"), bg="#d1e7dd")
sentiment_label.pack(pady=10)

# Summary Frame
summary_frame = tk.Frame(window, bg="#fff3cd", bd=2, relief="ridge")
summary_frame.pack(pady=10, padx=20, fill="both", expand=True)
summary_label = tk.Label(summary_frame, text="Summary:", font=("Arial", 14, "bold"), bg="#fff3cd")
summary_label.pack(pady=5)
summary_box = scrolledtext.ScrolledText(summary_frame, height=8, width=95, font=("Arial", 12), state='disabled')
summary_box.pack(pady=5)

window.mainloop()
