# StudyMate – AI-Powered PDF Summarizer & Q/A Assistant

StudyMate is a web app built with Streamlit that allows you to:
- Upload PDFs (text-based)
- Get AI-generated summaries
- Ask questions and get relevant answers based on the PDF content

Powered by:
- Hugging Face Transformers (BART-large for summarization)
- Sentence Transformers (semantic search for Q/A)
- PyMuPDF for PDF text extraction

---

## Features
- Interactive UI with a modern theme
- Typing animation for the title
- Multi-page navigation
- Semantic search to find relevant answers
- Summarization of large documents in chunks
- Works locally after models are downloaded

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/studymate.git
cd studymate

→Create a virtual environment and activate it
  python -m venv venv
  # Windows
  venv\Scripts\activate
  # Mac/Linux
  source venv/bin/activate
→Install dependencies
 pip install -r requirements.txt
→Requirements
 Your requirements.txt should contain:
 streamlit
 PyMuPDF
 transformers
 sentence-transformers
 huggingface_hub
→Hugging Face Authentication
 Set your Hugging Face token in the environment variable:
 export HF_TOKEN=your_huggingface_token
 #Windows (PowerShell)
 setx HF_TOKEN "your_huggingface_token"
 Get your token from: https://huggingface.co/settings/tokens
 Usage
→Run the app locally:
 streamlit run app.py
