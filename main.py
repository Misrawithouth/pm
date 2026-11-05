import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google.generativeai import configure, embed_content, GenerativeModel
from pypdf import PdfReader

# Configuration from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="P&M Technical Support High-Speed Backend")

# Enable CORS for frontend integration (GitHub Pages, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB (Local persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pm_manuals")

class QueryRequest(BaseModel):
    query: str
    machineKey: str = None
    isGeneral: bool = False
    lang: str = "en"

def get_gemini_embedding(text: str):
    response = embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

@app.get("/")
def health_check():
    return {"status": "operational", "system": "P&M FastAPI + ChromaDB Gateway"}

@app.post("/api/ask")
def ask_support(req: QueryRequest):
    try:
        raw_query = req.query.strip()
        lang_mode = req.lang.lower()
        
        # 1. Generate Query Embedding for ChromaDB search
        query_emb_resp = embed_content(
            model="models/text-embedding-004",
            content=raw_query,
            task_type="retrieval_query"
        )
        query_vector = query_emb_resp['embedding']
        
        # 2. ChromaDB Similarity Search (Ultra-Fast)
        where_filter = {"machine": req.machineKey.lower()} if req.machineKey else None
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
            where=where_filter if req.machineKey else None
        )
        
        context_text = "\n--- RETRIEVED MANUAL SECTIONS ---\n"
        source_cards = []
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i, doc in enumerate(docs):
                meta = metas[i]
                score = max(0, int((1.0 - distances[i]) * 100)) # Convert distance to rough score
                context_text += f"[Source {i+1}: {meta.get('fileName')} | Page {meta.get('pageNumber')} | Match: {score}%]:\n{doc}\n\n"
                
                source_cards.append({
                    "fileName": meta.get('fileName'),
                    "page": meta.get('pageNumber'),
                    "relevance": score,
                    "digest": doc[:110] + "..."
                })
        else:
            context_text += "No direct manual chunk matched. Relying on site engineering knowledge base.\n"

        # 3. Generate response using Gemini Flash Model
        prompt = (
            "YOU ARE A LEAD PLANT & MACHINERY (P&M) SPECIALIST CHIEF ENGINEER AT SITE.\n"
            "MANDATE: Provide a complete step-by-step fluid or electrical circuit flow walkthrough, physical mechanism, and exact schematic port references without cutting off.\n"
            f"LANGUAGE: {'Formal Hindi (Devanagari)' if lang_mode=='hi' else ('Workshop Hinglish' if lang_mode=='hinglish' else 'Technical English')}.\n"
            f"RETRIEVED MANUAL DATA:\n{context_text}\n\n"
            f"Engineer Query: {raw_query}"
        )
        
        model = GenerativeModel("gemini-2.5-flash") # Or latest working flash model
        ai_resp = model.generate_content(prompt)
        reply_text = ai_resp.text.strip() if ai_resp and ai_resp.text else "System active. Check manual flow."

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
    customTitle: str = Form(None),
    categoryDesc: str = Form("General Equipment")
):
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Read PDF and extract text page by page
        reader = PdfReader(file_path)
        vectors_to_add = []
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
                
            # Chunking text for precise vector search
            chunk_size = 600
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) < 10:
                    continue
                    
                chunk_id = f"{file_id}_p{page_num+1}_c{i}"
                emb = get_gemini_embedding(chunk)
                
                documents.append(chunk)
                metadatas.append({
                    "fileId": file_id,
                    "fileName": file_name,
                    "pageNumber": page_num + 1,
                    "machine": categoryDesc.lower().strip()
                })
                ids.append(chunk_id)
                embeddings.append(emb)
                
        # Batch insert into ChromaDB
        if ids:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        # Cleanup temp file
        os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"Successfully processed {file_name} and indexed into ChromaDB!"
        }
    except Exception as e:
        return {"status": "error", "message": f"Upload indexing failed: {str(e)}"}
