from utils.resume_parser import extract_info    
from utils.job_matcher import match_jobs  
from utils.ai_suggester import get_ai_advice 
from utils.cleaner import clean_resume_text 
import streamlit as st 
import pdfplumber 
import spacy

# Load the English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer & Job Matcher")

uploaded = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        text = "".join([page.extract_text() or "" for page in pdf.pages])
    st.subheader("📃 Resume Preview")
    st.text(text[:1000])

    cleaned_text = clean_resume_text(text)
    info = extract_info(cleaned_text)

    st.subheader("🧠 Extracted Skills & Info")
    st.json(info)

    jobs = match_jobs(info["skills"])
    st.subheader("🎯 Matched Jobs")
    st.dataframe(jobs)

    st.subheader("🤖 AI Career Suggestions")
    suggestion = get_ai_advice(info)
    st.markdown(suggestion)