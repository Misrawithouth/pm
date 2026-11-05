import os
import shutil
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google.generativeai import configure, embed_content, GenerativeModel
from pypdf import PdfReader

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="P&M Technical Support High-Speed Backend")

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
    isGeneral: Optional[bool] = False
    lang: Optional[str] = "hinglish"

def safe_gemini_call(prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            model = GenerativeModel("gemini-3.8-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "quota" in err_str.lower()) and attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            elif attempt == retries - 1:
                return f"System busy hai. Kripya thoda wait karke dobara try karein."
    return "System operational."

@app.get("/")
def health_check():
    return {"status": "operational", "system": "P&M FastAPI + ChromaDB Gateway"}

@app.post("/api/ask")
def ask_support(req: QueryRequest):
    try:
        raw_query = (req.query or "").strip()
        lang_mode = (req.lang or "hinglish").lower()
        
        # Check if query is a greeting or casual expression
        greetings = ["hi", "hello", "hey", "kaise ho", "gm", "good morning", "sup", "namaste"]
        is_greeting = raw_query.lower() in greetings or len(raw_query.split()) <= 2 and any(g in raw_query.lower() for g in ["hi", "hello", "hey"])

        context_text = "--- SITE TECHNICAL KNOWLEDGE BASE ---\n"
        source_cards = []
        
        if not is_greeting and GEMINI_API_KEY:
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

        if is_greeting:
            prompt = (
                "You are a friendly and helpful Plant & Machinery (P&M) Assistant at a site. "
                "Respond naturally and warmly to the user's greeting in conversational Workshop Hinglish."
            )
        elif req.machineKey and len(source_cards) == 0:
            # STRICT MANUAL-ONLY RULE ENFORCEMENT
            return {
                "status": "success",
                "reply": f"❌ Yeh jaankari is machine ke uploaded manual mein uplabdh nahi hai.",
                "sources": [],
                "followUps": []
            }
        else:
            prompt = (
                "YOU ARE A LEAD PLANT & MACHINERY (P&M) SPECIALIST CHIEF ENGINEER AT SITE.\n"
                "MANDATE: Provide a complete step-by-step fluid or electrical circuit flow walkthrough, physical mechanism, and exact schematic port references based strictly on the retrieved manual data.\n"
                f"LANGUAGE: {'Formal Hindi (Devanagari)' if lang_mode=='hi' else ('Workshop Hinglish' if lang_mode=='hinglish' else 'Technical English')}.\n"
                f"RETRIEVED MANUAL DATA:\n{context_text}\n\n"
                f"Engineer Query: {raw_query}"
            )
        
        reply_text = safe_gemini_call(prompt) if GEMINI_API_KEY else "System active without API key."

        return {
            "status": "success",
            "reply": reply_text,
            "sources": source_cards,
            "followUps": ["Check Main Relief Valve", "Inspect Return Circuit", "View Schematic Drawing"]
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
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        file_id = f"file_{os.path.basename(file_path)}_{os.urandom(4).hex()}"
        file_name = customTitle or file.filename
        
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
                        emb_resp = embed_content(
                            model="models/text-embedding-004",
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        emb = emb_resp['embedding']
                    except Exception:
                        pass
                
                documents.append(chunk)
                metadatas.append({
                    "fileId": file_id,
                    "fileName": file_name,
                    "pageNumber": page_num + 1,
                    "machine": (categoryDesc or "General").lower().strip()
                })
                ids.append(chunk_id)
                embeddings.append(emb)
                
        if ids and GEMINI_API_KEY:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"Successfully processed {file_name} and indexed into ChromaDB!"
        }
    except Exception as e:
        return {"status": "error", "message": f"Upload indexing failed: {str(e)}"}
