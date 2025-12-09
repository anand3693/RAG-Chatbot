# RAG Document Chat System (FastAPI + LangChain)

An AI-powered document question-answering system using **Retrieval Augmented Generation (RAG)**.  
Upload PDFs, text files, and scanned images — the system extracts content using OCR (Tesseract), stores embeddings in FAISS, and enables **conversational queries** grounded in your documents.

---

## 🚀 Features

🔹 Upload and process multiple document formats (PDF, TXT, and Images)  
🔹 OCR support for scanned PDFs and images  
🔹 Streaming responses for faster interaction  
🔹 Chat memory — maintains context across conversation  
🔹 FAISS vector search for accurate retrieval  
🔹 Uses open-source Hugging Face models (no paid API required)  
🔹 Clear separation between UI, backend logic, and vector store  
🔹 Option to reset chat and document knowledge base anytime

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | FastAPI |
| RAG | LangChain |
| Embeddings | HuggingFace |
| Vector DB | FAISS |
| OCR | Pytesseract |
| File Handling | Python, UUID |
| Model | FLAN-T5 or compatible HF model |

---

## 📂 Project Structure

📦 chatbot
├─ data/                  # Predefined knowledge base files (if any)
├─ faiss_db/              # Vector database storage (auto-generated)
├─ static/                # Static assets (UI files, CSS, JS) if used
├─ uploaded_files/        # User-uploaded files stored here
├─ venv/                  # Python virtual environment (excluded in Git)
├─ __pycache__/           # Python cache files (excluded in Git)
├─ main.py                # FastAPI application code
├─ requirements.txt       # Python dependencies
└─ README.md              # Documentation (will be added)


1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate (Mac/Linux)

2. Install Dependencies
pip install -r requirements.txt

3. Add Hugging Face Token

Create .env file:

HF_TOKEN=your_huggingface_access_token

4. Run the App
uvicorn main:app --reload
