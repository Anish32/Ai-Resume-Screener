import streamlit as st
import os
import fitz  # PyMuPDF
from collections import Counter
from google.colab import files

# Step 1: Upload Resume
uploaded = files.upload()

# Step 2: Load and extract PDF text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Step 3: Preprocess text (basic)
def preprocess_text(text):
    return text.lower().replace('\n', ' ').replace('\r', ' ').strip()

# Step 4: Define Job Description Keywords
job_description = """
We are looking for a Machine Learning Engineer with experience in Python, data analysis, 
TensorFlow or PyTorch, and experience working with large language models (LLMs).
Knowledge of NLP, Scikit-learn, SQL, cloud platforms (AWS or GCP), and Git is a plus.
"""

job_keywords = [
    "machine learning", "python", "data analysis", "tensorflow", "pytorch",
    "llm", "nlp", "scikit-learn", "sql", "aws", "gcp", "git"
]

# Step 5: Score Resume against keywords
def score_resume(resume_text, keywords):
    score = 0
    found = []
    missing = []

    for keyword in keywords:
        if keyword in resume_text:
            found.append(keyword)
            score += 1
        else:
            missing.append(keyword)
    
    return {
        "score": score,
        "total": len(keywords),
        "percentage": round((score / len(keywords)) * 100, 2),
        "found_keywords": found,
        "missing_keywords": missing
    }

# Step 6: Run ATS Checker
for file_name in uploaded.keys():
    resume_raw = extract_text_from_pdf(file_name)
    resume_clean = preprocess_text(resume_raw)
    results = score_resume(resume_clean, job_keywords)

    # Step 7: Print Report
    print(f"\n--- ATS Resume Check for: {file_name} ---")
    print(f"Score: {results['score']} / {results['total']} ({results['percentage']}%)")
    print(f"✅ Found Keywords: {', '.join(results['found_keywords'])}")
    print(f"❌ Missing Keywords: {', '.join(results['missing_keywords'])}")
