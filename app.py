import os
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

# Hugging Face authentication
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
        device=-1  # CPU
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, embedder

summarizer, embedder = load_models()

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text.strip()

# Text chunking
def split_text_into_chunks(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# Summarization
def summarize_large_text(text):
    if len(text.split()) < 50:
        return "Document too short to summarize."
    summaries = []
    for chunk in split_text_into_chunks(text, max_words=400):
        try:
            summary_chunk = summarizer(
                chunk,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary_chunk)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {e}]")
    return " ".join(summaries)

# Question answering
def answer_question(question, text):
    sentences = text.split(". ")
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, sentence_embeddings, top_k=5)[0]
    answers = [sentences[hit['corpus_id']] for hit in hits if hit['score'] > 0.3]
    return " ".join(answers) if answers else "No relevant answer found."

# ==================== FRONTEND ====================

st.set_page_config(page_title="📘 StudyMate - PDF Summarizer & Q/A", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
        .main { padding: 20px; }
        .title { font-size: 38px; font-weight: bold; text-align: center; }
        .subtitle { text-align: center; font-size: 18px; opacity: 0.8; }
        .footer { text-align: center; padding: 15px; margin-top: 30px; opacity: 0.6; font-size: 14px; }
        .stButton>button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            border-radius: 8px;
            padding: 10px 18px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu
menu = st.sidebar.radio(
    "📑 Menu",
    ["🏠 Home", "📂 Upload & Preview", "📝 Summarize", "💬 Ask a Question", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Upload a clear, text-based PDF for the best results.")
st.sidebar.markdown("**Version:** 2.0 🚀")

# Home Page
if menu == "🏠 Home":
    st.markdown("<div class='title'>📘 StudyMate</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your AI-powered PDF Summarizer & Q/A Assistant</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Welcome to **StudyMate**, where you can upload large PDFs, get AI-generated summaries, and ask intelligent questions based on the content.")
    st.write("Use the **menu** on the left to navigate through features.")

# Upload & Preview
elif menu == "📂 Upload & Preview":
    uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.markdown("### 📄 PDF Text Preview")
        st.info(pdf_text[:1200] + "..." if len(pdf_text) > 1200 else pdf_text)
        st.session_state["pdf_text"] = pdf_text

# Summarize PDF
elif menu == "📝 Summarize":
    if "pdf_text" in st.session_state:
        if st.button("📝 Generate Summary"):
            with st.spinner("Summarizing large document... Please wait..."):
                summary = summarize_large_text(st.session_state["pdf_text"])
            st.markdown("### 📝 Summary")
            st.success(summary)
            st.session_state["summary"] = summary
    else:
        st.warning("⚠ Please upload a PDF first from the 'Upload & Preview' section.")

# Ask Question
elif menu == "💬 Ask a Question":
    if "pdf_text" in st.session_state:
        user_question = st.text_input("💬 Type your question here:")
        if st.button("🔍 Get Answer"):
            with st.spinner("Searching for the best answer..."):
                answer = answer_question(user_question, st.session_state["pdf_text"])
            st.markdown("### 📢 Answer")
            st.success(answer)
    else:
        st.warning("⚠ Please upload a PDF first from the 'Upload & Preview' section.")

# About Section
elif menu == "ℹ️ About":
    st.markdown("## ℹ️ About StudyMate")
    st.write("""
    StudyMate is an AI-powered tool that:
    - 📂 Extracts text from PDFs
    - 📝 Summarizes large documents
    - 💬 Answers questions based on PDF content
    
    Built using **Streamlit**, **Hugging Face Transformers**, and **Sentence Transformers**.
    """)
    st.markdown("#### 👨‍💻 Developer")
    st.write("Built with ❤️ by Your Name Here.")

# Footer
st.markdown("---")
st.markdown("<div class='footer'>🚀 Built with ❤️ using Streamlit | StudyMate 2025</div>", unsafe_allow_html=True)
