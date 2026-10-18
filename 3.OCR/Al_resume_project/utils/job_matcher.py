import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_jobs(resume_skills):
    df = pd.read_csv("jobs.csv")

    # Debug print (optional)
    print("CSV Columns:", df.columns.tolist())

    # Ensure the columns exist
    if "Title" not in df.columns or "Skills" not in df.columns:
        raise ValueError("CSV must contain 'Title' and 'Skills' columns.")

    df["combined"] = df["Title"] + " " + df["Skills"]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"])

    resume_text = " ".join(resume_skills)
    resume_vector = vectorizer.transform([resume_text])

    similarity_scores = cosine_similarity(resume_vector, tfidf_matrix).flatten()
    df["score"] = similarity_scores

    matched = df.sort_values(by="score", ascending=False).head(5)
    return matched[["Title", "Skills", "score"]]
