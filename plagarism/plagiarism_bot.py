# plagiarism_bot.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# Initialize semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Helper Functions ------------------ #

def read_txt(file_path):
    """Read text from a .txt file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text):
    """Lowercase, remove punctuation, remove stopwords"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

def tfidf_similarity(text1, text2):
    """Compute TF-IDF cosine similarity"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def semantic_similarity(text1, text2):
    """Compute semantic similarity using sentence embeddings"""
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return float(cosine_scores.max())

# ------------------ Main Bot Function ------------------ #

def plagiarism_check(file1, file2):
    raw_text1 = read_txt(file1)
    raw_text2 = read_txt(file2)

    text1 = preprocess_text(raw_text1)
    text2 = preprocess_text(raw_text2)

    tfidf_score = tfidf_similarity(text1, text2)
    semantic_score = semantic_similarity(raw_text1, raw_text2)

    print("\n------ Plagiarism Check Result ------")
    print(f"TF-IDF Similarity: {tfidf_score*100:.2f}%")
    print(f"Semantic Similarity (paraphrasing): {semantic_score*100:.2f}%")

    threshold = 0.7
    if tfidf_score > threshold or semantic_score > threshold:
        print("\n⚠️ Potential plagiarism detected!")
    else:
        print("\n✅ Texts are sufficiently different.")

# ------------------ CLI Interaction ------------------ #

if __name__ == "__main__":
    print("=== Text-Only Plagiarism Check Bot ===")
    file1 = input("Enter path to first text file: ")
    file2 = input("Enter path to second text file: ")
    plagiarism_check(file1, file2)

def main():
    from plagiarism_bot import run_cli  # or rename plagiarism_check CLI function
    run_cli()

if __name__ == "__main__":
    main()
