import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Optional, List
import fitz  # PyMuPDF
import zipfile
import tempfile
import os
import json
import re
from dotenv import load_dotenv

# -------------------- ENV SETUP --------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

# -------------------- GEMINI MODEL --------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -------------------- DATA SCHEMA --------------------
class DataFormat(TypedDict, total=False):
    summary: Optional[str]
    skills: Optional[List[str]]
    experience: Optional[int]

# -------------------- PROMPT --------------------
PROMPT_TEMPLATE = """
Extract the following information from the resume text.
Return ONLY valid JSON.

Fields:
- summary: short professional summary
- skills: list of technical skills
- experience: total years of experience as an integer

If a field is missing, return null.

Resume Text:
\"\"\"{text}\"\"\"
"""

# -------------------- PDF TEXT EXTRACTION --------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# -------------------- RESUME PARSER --------------------
def parse_resume(text: str) -> DataFormat:
    try:
        response = model.invoke(PROMPT_TEMPLATE.format(text=text))
        response_text = response.content.strip()

        # Remove markdown JSON formatting if present
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        return json.loads(response_text)

    except Exception as e:
        return {"error": str(e)}

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Resume ZIP Parser", layout="wide")
st.title("📄 Resume ZIP → Structured JSON Parser")

st.markdown(
    """
Upload a **ZIP file containing PDF resumes**.  
Each resume will be parsed into a structured JSON format using **Gemini LLM**.
"""
)

uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])

# -------------------- PROCESS ZIP --------------------
if uploaded_zip:
    st.success("ZIP file uploaded successfully!")

    if st.button("🚀 Parse Resumes"):
        with st.spinner("Processing resumes..."):
            results = {}

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "resumes.zip")

                # Save uploaded ZIP
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())

                # Extract ZIP
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)

                # Walk through all folders to find PDFs
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.lower().endswith(".pdf"):
                            pdf_path = os.path.join(root, file)

                            try:
                                resume_text = extract_text_from_pdf(pdf_path)
                                parsed_output = parse_resume(resume_text)
                                results[file] = parsed_output
                            except Exception as e:
                                results[file] = {"error": str(e)}

            # -------------------- OUTPUT --------------------
            st.success(f"✅ Parsed {len(results)} resumes")

            for filename, data in results.items():
                st.subheader(filename)
                st.json(data)

            st.download_button(
                label="⬇ Download All Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name="parsed_resumes.json",
                mime="application/json"
            )