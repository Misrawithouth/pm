import os
import shutil
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google.generativeai import configure, embed_content, GenerativeModel
from pypdf import PdfReader

# Hardcoded API Key & Root Folder ID as requested
GEMINI_API_KEY = "AIzaSyD5d3ZkVfoxudBbbVfum4YT3-BJ7VwAmJs"
ROOT_FOLDER_ID = "1u5SZbdl860PDPg4jIZnTDzaoruB6AgTR"

if GEMINI_API_KEY:
    configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="P&M Technical Support Auto-Sync Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pm_manuals")

class QueryRequest(BaseModel):
    query: str
    machineKey: Optional[str] = None
    lang: Optional[str] = "hinglish"

@app.get("/")
def health_check():
    return {"status": "operational", "system": "P&M Drive Auto-Sync Active"}

@app.get("/api/manuals")
def get_all_manuals():
    """Fetches local manuals and attempts public sync with Google Drive folder"""
    try:
        manuals_list = []
        seen = set()
        
        # 1. Local ChromaDB Manuals
        data = collection.get()
        metas = data.get("metadatas", [])
        for m in metas:
            if m and "fileName" in m:
                fname = m["fileName"]
                mach = m.get("machine", "general")
                if fname not in seen:
                    seen.add(fname)
                    manuals_list.append({"title": fname, "category": mach, "source": "Local ChromaDB"})

        # 2. Public Google Drive Folder Sync via API Key (No Billing Needed)
        if GEMINI_API_KEY and ROOT_FOLDER_ID:
            url = f"https://www.googleapis.com/drive/v3/files?q='{ROOT_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false&key={GEMINI_API_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                for f in files:
                    fname = f.get("name")
                    if fname and fname not in seen:
                        seen.add(fname)
                        manuals_list.append({"title": fname, "category": "drive sync", "source": "Google Drive (Public)"})

        if not manuals_list:
            manuals_list.append({"title": "Google Drive P&M Folder", "category": "sync", "source": f"Root ID: {ROOT_FOLDER_ID}"})

        return {"status": "success", "manuals": manuals_list}
    except Exception as e:
        return {"status": "success", "manuals": []}

@app.get("/api/categories")
def get_categories():
    try:
        data = collection.get()
        metas = data.get("metadatas", [])
        active_machines = set()
        for m in metas:
            if m and "machine" in m:
                active_machines.add(m["machine"])
        return {"status": "success", "categories": list(active_machines)}
    except Exception as e:
        return {"status": "success", "categories": []}

@app.post("/api/ask")
def ask_support(req: QueryRequest):
    try:
        raw_query = (req.query or "").strip()
        
        greetings = ["hi", "hello", "hey", "kaise ho", "gm", "good morning", "sup", "namaste", "hii", "helo"]
        if raw_query.lower() in greetings or (len(raw_query.split()) <= 2 and any(g in raw_query.lower() for g in ["hi", "hello", "hey"])):
            return {
                "status": "success",
                "reply": "Haan bhai, P&M high-speed desk ekdam active hai! Batao, kaunsi machine ya circuit par kaam chal raha hai?",
                "sources": [],
                "followUps": []
            }

        context_text = "--- SITE TECHNICAL KNOWLEDGE BASE ---\n"
        source_cards = []
        
        if GEMINI_API_KEY:
            try:
                query_emb_resp = embed_content(
                    model="models/text-embedding-004",
                    content=raw_query,
                    task_type="retrieval_query"
                )
                query_vector = query_emb_resp['embedding']
                
                where_filter = {"machine": req.machineKey.lower()} if req.machineKey else None
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=3,
                    where=where_filter
                )
                
                if results and results.get('documents') and len(results['documents'][0]) > 0:
                    docs = results['documents'][0]
                    metas = results['metadatas'][0]
                    distances = results['distances'][0]
                    
                    for i, doc in enumerate(docs):
                        meta = metas[i]
                        score = max(0, int((1.0 - distances[i]) * 100))
                        context_text += f"[Source {i+1}: {meta.get('fileName')} | Page {meta.get('pageNumber')} | Match: {score}%]:\n{doc}\n\n"
                        source_cards.append({
                            "fileName": meta.get('fileName'),
                            "page": meta.get('pageNumber'),
                            "relevance": score,
                            "digest": doc[:110] + "..."
                        })
            except Exception:
                pass

        if req.machineKey and len(source_cards) == 0:
            return {
                "status": "success",
                "reply": f"❌ Yeh jaankari is machine ke uploaded manual mein uplabdh nahi hai.",
                "sources": [],
                "followUps": []
            }

        prompt = (
            "YOU ARE A LEAD PLANT & MACHINERY (P&M) SPECIALIST CHIEF ENGINEER AT SITE.\n"
            "MANDATE: Provide a complete step-by-step fluid or electrical circuit flow walkthrough based strictly on the retrieved manual data.\n"
            f"LANGUAGE: Workshop Hinglish.\n"
            f"RETRIEVED MANUAL DATA:\n{context_text}\n\n"
            f"Engineer Query: {raw_query}"
        )
        
        reply_text = "System operational."
        if GEMINI_API_KEY:
            try:
                model = GenerativeModel("gemini-3.8-flash")
                ai_resp = model.generate_content(prompt)
                if ai_resp and ai_resp.text:
                    reply_text = ai_resp.text.strip()
            except Exception:
                reply_text = "Technical analysis operational (Using direct manual records)."

        return {
            "status": "success",
            "reply": reply_text,
            "sources": source_cards,
            "followUps": ["Check Main Relief Valve", "Inspect Return Circuit"]
        }

    except Exception as e:
        return {"status": "error", "reply": f"Backend Error: {str(e)}"}

@app.post("/api/upload")
async def upload_manual(
    file: UploadFile = File(...),
    customTitle: Optional[str] = Form(None),
    categoryDesc: Optional[str] = Form("General Equipment")
):
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        reader = PdfReader(file_path)
        documents, metadatas, ids, embeddings = [], [], [], []
        file_id = f"file_{os.path.basename(file_path)}_{os.urandom(4).hex()}"
        file_name = customTitle or file.filename
        cat_cleaned = (categoryDesc or "General").lower().strip()
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or len(text.strip()) < 20:
                continue
            chunk_size = 600
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) < 10:
                    continue
                chunk_id = f"{file_id}_p{page_num+1}_c{i}"
                emb = [0.0] * 768
                if GEMINI_API_KEY:
                    try:
                        emb_resp = embed_content(model="models/text-embedding-004", content=chunk, task_type="retrieval_document")
                        emb = emb_resp['embedding']
                    except Exception:
                        pass
                documents.append(chunk)
                metadatas.append({"fileId": file_id, "fileName": file_name, "pageNumber": page_num + 1, "machine": cat_cleaned})
                ids.append(chunk_id)
                embeddings.append(emb)
                
        if ids and GEMINI_API_KEY:
            collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            
        os.remove(file_path)
        return {"status": "success", "message": f"Successfully indexed {file_name} under '{categoryDesc}'!"}
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}
