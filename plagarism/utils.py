# utils.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ---------------- NLTK Setup ---------------- #
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# ---------------- Semantic Model ---------------- #
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- Helper Functions ---------------- #

def read_txt(file_path: str) -> str:
    """
    Read text content from a .txt file.
    
    Args:
        file_path (str): Path to the text file.
    
    Returns:
        str: File content as string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing, removing punctuation and stopwords.
    
    Args:
        text (str): Raw text.
    
    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

def tfidf_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.
    
    Args:
        text1 (str): First text.
        text2 (str): Second text.
    
    Returns:
        float: Similarity score (0 to 1).
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using sentence embeddings.
    
    Args:
        text1 (str): First text.
        text2 (str): Second text.
    
    Returns:
        float: Maximum similarity score between sentences (0 to 1).
    """
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return float(cosine_scores.max())
