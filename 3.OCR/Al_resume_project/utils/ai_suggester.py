import google.generativeai as genai
import os
from google.generativeai import configure

# =======================
# Load environment variables from .env
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_GENAI_API_KEY")
genai.configure(api_key=api_key)

# ===============
# for github action token setup
# genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))


# =========================

# Configure API key from Streamlit secrets
# genai.configure(api_key=st.secrets["GOOGLE_GENAI_API_KEY"])

def get_ai_advice(info: dict) -> str:
    prompt = f"""
    I'm sharing the following resume information:

    Skills:
    {', '.join(info.get('skills', []))}

    Entities / Experience:
    {', '.join(info.get('entities', []))}

    Based on the above, give me AI-generated career suggestions.
    """
    
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
