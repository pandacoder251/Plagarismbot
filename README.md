# 🕵️‍♂️Plagiarism Check Bot

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/plagiarism_bot?style=flat-square)](https://github.com/yourusername/plagiarism_bot/issues)
[![Last Commit](https://img.shields.io/github/last-commit/yourusername/plagiarism_bot?style=flat-square&color=yellow)]()
[![Repo Size](https://img.shields.io/github/repo-size/yourusername/plagiarism_bot?style=flat-square&color=orange)]()

A **Python-based plagiarism checker** that compares two text files, detects both **copy-paste and paraphrased content**, and flags potential plagiarism.  
Ideal for students, educators, and developers seeking a **fast and reliable text similarity tool**.

---

## ✨ Features

- Compare **two `.txt` files**
- Detect **direct copy-paste plagiarism** using **TF-IDF similarity**
- Identify **paraphrased content** via **semantic similarity (sentence embeddings)**
- Customizable threshold (default **70%**)
- **Interactive CLI** for a user-friendly experience
- Modular and extensible codebase

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Libraries:** `nltk`, `scikit-learn`, `sentence-transformers`
- **Optional:** GUI (`tkinter`) or Web App (`streamlit`)

---

## 📁 Project Structure

```
User Input
    │
    ▼
Read Text Files ──> Error Handling
    │
    ▼
Preprocess Text (TF-IDF)
    │
    ▼
Compute TF-IDF Similarity
Compute Semantic Similarity
    │
    ▼
Compare to Threshold
    │
    ▼
Display Results in CLI
    │
    ▼
Optional: Generate Report (Future)

```

---

## 📊 Output

The bot provides:
- TF-IDF similarity (%)
- Semantic similarity (%)
- Plagiarism alert (if threshold exceeded)

---

## 👩‍💻 About the Developer

**Pousali Dolai**  
2nd-year B.Tech CSE student @ ITER SOA  
Passionate about **web development** & **sustainable tech 🌍**

### 💼 Core Skills
- Frontend: HTML, CSS, JavaScript, Tailwind CSS
- Python, Java
- Content Writing & Blogging
- Team Collaboration

### 🚀 Highlights
- Built **10+ frontend projects**
- Currently learning: **Three.js, Next.js, DSA, Python, AI prompts**
- Goal: Master **Full Stack Development** and explore **AI Design**

---

