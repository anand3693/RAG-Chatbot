import os
import shutil
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import traceback


from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDEFINED_DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADED_DATA_DIR = os.path.join(BASE_DIR, "uploaded_files")
DB_DIR = os.path.join(BASE_DIR, "faiss_db")

os.makedirs(PREDEFINED_DATA_DIR, exist_ok=True)
os.makedirs(UPLOADED_DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_chain = None
memory = None

def get_llm():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=1024,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=generator)

def ocr_image(path):
    try:
        image = Image.open(path).convert("L")
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        text = pytesseract.image_to_string(image)
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text
    except Exception as e:
        print(f"OCR error for {path}: {e}")
        return ""

def load_file(path):
    documents = []
    filename = os.path.basename(path)

    try:
        lower = filename.lower()
        if lower.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif lower.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            text = ocr_image(path)
            if not text:
                return []
            txt_path = path + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            loader = TextLoader(txt_path, encoding="utf-8")
        else:
            return []

        loaded_docs = loader.load()

        for d in loaded_docs:
            if not hasattr(d, "metadata") or d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = filename

        documents.extend(loaded_docs)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        traceback.print_exc()

    return documents

def load_documents():
    documents = []

    for file in os.listdir(PREDEFINED_DATA_DIR):
        documents.extend(load_file(os.path.join(PREDEFINED_DATA_DIR, file)))

    for file in os.listdir(UPLOADED_DATA_DIR):
        documents.extend(load_file(os.path.join(UPLOADED_DATA_DIR, file)))

    print(f"Loaded {len(documents)} documents")
    return documents

CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant that provides accurate information based ONLY on the given context. 

IMPORTANT RULES:
1. If the information is not in the context, say "I don't have information about that in the provided documents."
2. Do NOT make up or invent any information that is not in the context.
3. Be specific and comprehensive when answering about lists, skills, experiences, etc.
4. Extract and present ALL relevant information from the context.

Context: {context}

Chat History: {chat_history}

Question: {question}

If the answer is not in the context, say "I don't have information about that in the provided documents."
Otherwise, provide a comprehensive answer based ONLY on the context:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)

def create_enhanced_retriever(vectordb):
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 20,
            "lambda_mult": 0.6,
        }
    )

def initialize_memory():
    global memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    return memory

def create_rag_chain(force_rebuild_db: bool = False):
    global rag_chain, memory

    print(f"Creating RAG chain, force_rebuild_db: {force_rebuild_db}")

    if memory is None:
        memory = initialize_memory()

    if force_rebuild_db and os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)
            print("Removed existing FAISS DB for rebuild")
        except Exception as e:
            print(f"Failed to remove old DB: {e}")

    documents = load_documents()
    if not documents:
        print("No documents found to build RAG chain.")
        rag_chain = None
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    try:
        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(DB_DIR)
        print("FAISS index built and saved.")
    except Exception as e:
        print(f"Error building FAISS: {e}")
        traceback.print_exc()
        rag_chain = None
        return None

    retriever = create_enhanced_retriever(vectordb)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        verbose=False,
    )

    print("RAG chain created successfully")
    return rag_chain

def ensure_rag_chain():
    global rag_chain
    if rag_chain is not None:
        return rag_chain

    try:
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            print("Loading existing FAISS index...")
            vectordb = FAISS.load_local(DB_DIR, embeddings)
            retriever = create_enhanced_retriever(vectordb)
            
            if memory is None:
                memory = initialize_memory()
                
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=get_llm(),
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
                verbose=False,
            )
            print("Loaded existing FAISS and RAG chain.")
            return rag_chain
    except Exception as e:
        print(f"Could not load existing FAISS: {e}")

    print("Building new RAG chain...")
    return create_rag_chain(force_rebuild_db=True)

def analyze_retrieved_documents(source_docs, question):
    relevant_sources = []
    q_terms = set(question.lower().split())
    for doc in source_docs:
        source = (doc.metadata or {}).get("source", "unknown")
        content_terms = set((doc.page_content or "").lower().split())
        if len(q_terms.intersection(content_terms)) > 0:
            if source not in relevant_sources:
                relevant_sources.append(source)
    return relevant_sources

@app.on_event("startup")
async def startup_event():
    try:
        documents = load_documents()
        if not documents:
            print("No documents found at startup. Please upload files.")
        else:
            ensure_rag_chain()
    except Exception as e:
        print("Startup RAG build failed:", e)
        traceback.print_exc()
    print("Server started")

@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    return HTMLResponse(open(html_path, "r", encoding="utf-8").read())

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        file_path = os.path.join(UPLOADED_DATA_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        documents = load_file(file_path)
        if not documents:
            return {"filename": file.filename, "status": "failed", "chunks": 0}

        print(f"Uploaded {file.filename}: {len(documents)} documents")

        create_rag_chain(force_rebuild_db=True)

        return {"filename": file.filename, "status": "uploaded", "chunks": len(documents)}
    except Exception as e:
        print(f"Upload error: {e}")
        return {"filename": file.filename, "status": "error", "error": str(e)}

@app.post("/chat")
async def chat(message: str = Form(...)):
    global rag_chain

    user_msg = message.lower().strip()

    greetings = ["hi", "hello", "hey", "yo", "good morning", "good evening", "good afternoon"]
    if user_msg in greetings:
        return {"answer": "Hello! How can I help you today?"}

    if rag_chain is None:
        ensure_rag_chain()
    
    if rag_chain is None:
        return {"answer": "No documents available. Upload or place predefined data first."}

    try:
        enhanced_message = message
        list_keywords = ["list", "what are", "skills", "all the", "everything", "each", "every", "experience", "work history"]
        if any(keyword in user_msg for keyword in list_keywords):
            enhanced_message = f"{message} Please provide ALL relevant information found in the documents."

        result = rag_chain({"question": enhanced_message})
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        sources = []
        for doc in source_docs:
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source", "unknown")
            if src and src not in sources:
                sources.append(src)

        no_info_phrases = [
            "i don't have information",
            "not in the provided documents",
            "no information about",
            "not found in the context"
        ]
        has_relevant_info = not any(phrase in answer.lower() for phrase in no_info_phrases)

        if not has_relevant_info and sources:
            relevant_sources = analyze_retrieved_documents(source_docs, message)
            if relevant_sources:
                answer = f"I found some potentially relevant information in the documents, but couldn't extract a clear answer. The relevant documents are: {', '.join(relevant_sources)}. You might want to check these documents directly."
            else:
                answer = "I don't have information about that in the provided documents."

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"Error during chat: {e}")
        traceback.print_exc()
        rag_chain = None
        return {"answer": "I encountered an error while processing your request. Please try again.", "sources": []}

@app.get("/history")
async def get_history():
    if memory is None:
        return {"history": []}
    
    history_list = []
    for msg in memory.chat_memory.messages:
        history_list.append({
            "type": msg.type,
            "data": {"content": msg.content}
        })
    return {"history": history_list}

@app.post("/clear_history")
async def clear_history():
    global memory
    if memory:
        memory.clear()
    memory = initialize_memory()
    return {"status": "History cleared"}

@app.post("/new_chat")
async def new_chat():
    global rag_chain

    try:
        if memory:
            memory.clear()
    except Exception as e:
        print("Memory clear error:", e)

    rag_chain = None

    try:
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)
    except Exception as e:
        print("Error deleting DB_DIR:", e)

    try:
        for fname in os.listdir(UPLOADED_DATA_DIR):
            path = os.path.join(UPLOADED_DATA_DIR, fname)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Error deleting uploaded {fname}: {e}")
    except Exception as e:
        print("Error listing uploaded_files:", e)

    create_rag_chain(force_rebuild_db=True)

    return {"status": "New chat started — uploaded files cleared"}

@app.get("/debug/sources")
async def debug_sources():
    try:
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            return {"error": "FAISS DB not initialized or empty."}

        vectordb = FAISS.load_local(DB_DIR, embeddings)
        docs = []
        try:
            for key, doc in vectordb.docstore._dict.items():
                content = getattr(doc, "page_content", "")
                meta = getattr(doc, "metadata", {}) or {}
                docs.append({
                    "id": key,
                    "source": meta.get("source", "unknown"),
                    "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
                    "length": len(content)
                })
        except Exception:
            retrieved = vectordb.similarity_search("", k=100)
            for idx, d in enumerate(retrieved):
                docs.append({
                    "id": f"r{idx}",
                    "source": (d.metadata or {}).get("source", "unknown"),
                    "content_preview": (d.page_content or "")[:200],
                    "length": len(d.page_content or "")
                })

        return {"total_chunks": len(docs), "chunks": docs}

    except Exception as e:
        print("Debug sources error:", e)
        traceback.print_exc()
        return {"error": "Failed to read FAISS DB"}

@app.get("/health")
async def health_check():
    documents_exist = len(load_documents()) > 0
    rag_ready = rag_chain is not None
    db_exists = os.path.exists(DB_DIR) and os.listdir(DB_DIR)
    
    return {
        "status": "healthy",
        "documents_available": documents_exist,
        "rag_chain_ready": rag_ready,
        "faiss_db_exists": db_exists,
        "predefined_files": os.listdir(PREDEFINED_DATA_DIR),
        "uploaded_files": os.listdir(UPLOADED_DATA_DIR)
    }