To **use your Google Generative AI (GenAI) API key securely on GitHub for deploying a Streamlit project**, follow these steps:

---

### ✅ Step-by-Step Guide
## use python 3.13
---

#### 🔐 1. **Get Your Google GenAI API Key**

1. Go to [Google AI Studio](https://makersuite.google.com/app).
2. Sign in with your Google account.
3. Click **Get API Key**.
4. Copy the API key provided.

---

#### 📂 2. **Do NOT Hardcode the API Key**

Avoid adding your API key directly in your Python code (e.g., `api_key = "ABC123"`). This is **insecure**, especially if pushing code to GitHub.

---

#### 🔐 3. **Store API Key Securely in GitHub (For Public or Private Repo)**

##### a. If you're deploying from GitHub (e.g., via Streamlit Community Cloud):

1. Go to your GitHub repository.
2. Click **Settings** > **Secrets and variables** > **Actions** > **New repository secret**.
3. Add a new secret:

   * **Name**: `GOOGLE_GENAI_API_KEY`
   * **Value**: Paste your API key.
4. Save it.

---

#### 📦 4. **Access It in Python Code Using `os.environ`**

Install `os` if not already used:

```python
import os
from google.generativeai import configure
```

Access the secret like this:

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
```

---

#### 🗂️ 5. **Create `.streamlit/secrets.toml` (for Streamlit Cloud deployment)**

If you're using **Streamlit Community Cloud**, create the file:

```
.streamlit/secrets.toml
```

Content:

```toml
GOOGLE_GENAI_API_KEY = "your_api_key_here"
```

Then use it in Python like:

```python
import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_GENAI_API_KEY"])
```

---

### 🧪 Example: Using Google GenAI in Streamlit

```python
import streamlit as st
import google.generativeai as genai

# Configure API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_GENAI_API_KEY"])

# Load model
model = genai.GenerativeModel("gemini-pro")

st.title("Chat with Google GenAI (Gemini)")

user_input = st.text_input("You:", "Tell me a joke about AI")
if user_input:
    response = model.generate_content(user_input)
    st.write("Gemini:", response.text)
```

---

### 🛑 Important Tips:

* Never push your API key directly to GitHub.
* Use `.gitignore` to exclude any `.env` or config files with keys.
* For public deployments, always use `secrets.toml` or GitHub secrets.

---

Would you like a ready-made **Streamlit + Google GenAI GitHub project template**?
