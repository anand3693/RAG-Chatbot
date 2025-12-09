## 📘 RAG-Chatbot — Document-Aware AI Assistant

A Retrieval-Augmented Generation chatbot that allows users to upload PDFs, text files, and even images to extract knowledge and chat intelligently based on the uploaded content. Supports real-time Q&A, chat history, and dynamic UI — powered by FastAPI, LangChain, Hugging Face embeddings, and FAISS vector search.

<div align="center"> 🚀 *Ask questions from your documents. Get accurate, context-based answers instantly!* </div>
## 🌟 Key Features

✔ Upload PDFs, text files, scanned images
✔ Automatic OCR with text extraction
✔ FAISS-based vector search for relevant answers
✔ Real-time interactive chat UI
✔ Chat history view & management
✔ Reset chat + delete uploaded files
✔ Fully local processing (no cloud dependency)

## 🧩 Tech Stack
Backend

FastAPI

LangChain (RAG Pipeline)

FAISS (Vector Database)

Hugging Face Embeddings

OCR: Pytesseract + Pillow

Python-dotenv

Python 3.10+

Frontend

HTML5, CSS3, Teal UI Theme

JavaScript (Fetch API REST communication)

Scrollable real-time messaging UI

Storage
Purpose	Location
Uploaded files	uploaded_files/
Vector DB	faiss_db/
OAuth keys	.env


## 📂 Project Folder Structure
chatbot/
│
├── data/                 # Optional predefined documents
├── faiss_db/             # Vector store generated automatically
├── static/               # Frontend HTML, CSS, JS (UI)
├── uploaded_files/       # User uploads stored temporarily
│
├── main.py               # FastAPI backend + RAG chain implementation
├── requirements.txt      # Dependencies
├── .env                  # Hugging Face / secrets (Not included in git)
└── README.md

## 🛠 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/<your-username>/RAG-Chatbot.git
cd RAG-Chatbot

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add .env File

Create a .env file in project root:

HF_TOKEN=your_huggingface_api_key_here

(Get token from Hugging Face → Settings → Access Tokens)


## 📸 UI Preview

<img width="1890" height="896" alt="Screenshot (11)" src="https://github.com/user-attachments/assets/4df9cf90-4e07-4e57-af4e-110e68341892" />: Chat UI


<img width="1896" height="901" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/5a859c97-a79e-486b-8a76-77408226baac" />: History Sidebar


## 5️⃣ Run App
uvicorn main:app --reload
