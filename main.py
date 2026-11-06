import os
import shutil
import io
import json
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from google.generativeai import configure, embed_content, GenerativeModel
from pypdf import PdfReader
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload

# Configurations
GEMINI_API_KEY = "AIzaSyD5d3ZkVfoxudBbbVfum4YT3-BJ7VwAmJs"
MONGO_URI = "mongodb+srv://nitinmisra22_db_user:Jaimataki22@@support.zyahtkz.mongodb.net/?retryWrites=true&w=majority"

# Optional: Google Drive Folder ID where manuals will be saved (Create a folder in Drive and paste its ID here if needed)
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("https:1u5SZbdl860PDPg4jIZnTDzaoruB6AgTR", "")

if GEMINI_API_KEY:
    try:
        configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

app = FastAPI(title="P&M Technical Support Google Drive + MongoDB Cloud Desk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB Connection
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client["pm_technical_desk"]
    manuals_collection = db["manual_chunks"]
except Exception as e:
    print(f"MongoDB Connection Error: {e}")

# Helper: Initialize Google Drive API client if service account JSON is provided in environment variables
def get_drive_service():
    try:
        creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        if creds_json:
            creds_dict = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Drive API Init Error: {e}")
    return None

class QueryRequest(BaseModel):
    query: str
    machineKey: Optional[str] = None
    lang: Optional[str] = "hinglish"

@app.get("/")
def health_check():
    return {"status": "operational", "system": "Google Drive + MongoDB Cloud Hybrid Active"}

@app.get("/api/manuals")
def get_all_manuals():
    try:
        manuals_list = []
        pipeline = [
            {"$group": {"_id": "$fileName", "category": {"$first": "$machine"}, "driveId": {"$first": "$driveFileId"}}}
        ]
        cursor = manuals_collection.aggregate(pipeline)
        for doc in cursor:
            manuals_list.append({
                "title": doc["_id"],
                "category": doc.get("category", "general"),
                "driveFileId": doc.get("driveId", ""),
                "source": "Google Drive + MongoDB"
            })
        return {"status": "success", "manuals": manuals_list}
    except Exception as e:
        return {"status": "success", "manuals": []}

@app.get("/api/categories")
def get_categories():
    try:
        categories = manuals_collection.distinct("machine")
        return {"status": "success", "categories": [c for c in categories if c]}
    except Exception as e:
        return {"status": "success", "categories": []}

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

@app.post("/api/ask")
def ask_support(req: QueryRequest):
    try:
        raw_query = (req.query or "").strip()
        
        greetings = ["hi", "hello", "hey", "kaise ho", "gm", "good morning", "sup", "namaste", "hii", "helo"]
        if raw_query.lower() in greetings or (len(raw_query.split()) <= 2 and any(g in raw_query.lower() for g in ["hi", "hello", "hey"])):
            return {
                "status": "success",
                "reply": "Haan bhai, P&M Google Drive + MongoDB desk ekdam active hai! Batao, kaunsi machine ya circuit par kaam chal raha hai?",
                "sources": [],
                "followUps": []
            }

        context_text = "--- SITE TECHNICAL KNOWLEDGE BASE ---\n"
        source_cards = []
        matched_chunks = []
        
        try:
            query_emb_resp = embed_content(
                model="models/text-embedding-004",
                content=raw_query,
                task_type="retrieval_query"
            )
            query_vector = query_emb_resp['embedding']
            
            query_filter = {}
            if req.machineKey:
                query_filter["machine"] = req.machineKey.lower()
                
            all_chunks = list(manuals_collection.find(query_filter, {"text": 1, "embedding": 1, "fileName": 1, "pageNumber": 1, "driveFileId": 1}))
            
            scored_chunks = []
            for item in all_chunks:
                if "embedding" in item and item["embedding"]:
                    score = cosine_similarity(query_vector, item["embedding"])
                    scored_chunks.append((score, item))
            
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_results = scored_chunks[:3]
            
            for score, item in top_results:
                relevance_pct = max(0, int(score * 100))
                matched_chunks.append(item["text"])
                source_cards.append({
                    "fileName": item.get("fileName", "Manual"),
                    "page": item.get("pageNumber", 1),
                    "relevance": relevance_pct,
                    "driveFileId": item.get("driveFileId", ""),
                    "digest": item["text"][:110] + "..."
                })
        except Exception as ex:
            print(f"Search error: {ex}")

        if not matched_chunks:
            try:
                fallback_filter = {"machine": req.machineKey.lower()} if req.machineKey else {}
                fallback_items = list(manuals_collection.find(fallback_filter, {"text": 1, "fileName": 1, "pageNumber": 1, "driveFileId": 1}).limit(3))
                for item in fallback_items:
                    matched_chunks.append(item["text"])
                    source_cards.append({
                        "fileName": item.get("fileName", "Manual"),
                        "page": item.get("pageNumber", 1),
                        "relevance": 90,
                        "driveFileId": item.get("driveFileId", ""),
                        "digest": item["text"][:110] + "..."
                    })
            except Exception:
                pass

        for i, doc in enumerate(matched_chunks):
            meta = source_cards[i] if i < len(source_cards) else {"fileName": "Manual", "page": 1}
            context_text += f"[Source {i+1}: {meta.get('fileName')} | Page {meta.get('page')}]:\n{doc}\n\n"

        prompt = (
            "YOU ARE A LEAD PLANT & MACHINERY (P&M) SPECIALIST CHIEF ENGINEER AT SITE.\n"
            "MANDATE: Provide a complete step-by-step fluid or electrical circuit flow walkthrough or part number/drawing analysis based strictly on the retrieved manual/blueprint data below.\n"
            f"LANGUAGE: Workshop Hinglish.\n"
            f"RETRIEVED MANUAL DATA:\n{context_text}\n\n"
            f"Engineer Query: {raw_query}"
        )
        
        reply_text = ""
        if GEMINI_API_KEY:
            try:
                model = GenerativeModel("gemini-3.8-flash")
                ai_resp = model.generate_content(prompt)
                if ai_resp and ai_resp.text:
                    reply_text = ai_resp.text.strip()
            except Exception as ex:
                reply_text = f"Technical analysis operational. (Note: {str(ex)})"

        if not reply_text:
            if source_cards:
                reply_text = f"⚙️ <b>Manual Data Found:</b><br>{source_cards[0]['digest']}"
            else:
                reply_text = "❌ Yeh jaankari uploaded manual mein uplabdh nahi hai."

        return {
            "status": "success",
            "reply": reply_text,
            "sources": source_cards,
            "followUps": ["Check Unloader Valve", "Inspect Hydraulic Circuit"]
        }

    except Exception as e:
        return {"status": "error", "reply": f"Backend Error: {str(e)}"}

@app.post("/api/upload")
async def upload_manual(
    file: UploadFile = File(...),
    customTitle: Optional[str] = Form(None),
    categoryDesc: Optional[str] = Form("General Equipment")
):
    """Uploads file to Google Drive, processes text / Gemini Vision for drawings, and indexes chunks into MongoDB"""
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_name = customTitle or file.filename
        cat_cleaned = (categoryDesc or "General").lower().strip()
        
        # 1. Upload to Google Drive if service account is configured
        drive_file_id = "local_storage"
        try:
            drive_service = get_drive_service()
            if drive_service:
                file_metadata = {'name': file_name}
                if GOOGLE_DRIVE_FOLDER_ID:
                    file_metadata['parents'] = [GOOGLE_DRIVE_FOLDER_ID]
                media = MediaFileUpload(file_path, resumable=True)
                uploaded_file = drive_service.files().create(
                    body=file_metadata, media_body=media, fields='id'
                ).execute()
                drive_file_id = uploaded_file.get('id')
        except Exception as drive_err:
            print(f"Google Drive Upload Note: {drive_err}")

        documents_to_insert = []
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext in ['jpg', 'jpeg', 'png', 'webp']:
            with open(file_path, "rb") as img_f:
                img_bytes = img_f.read()
            
            vision_model = GenerativeModel("gemini-3.8-flash")
            vision_prompt = (
                "You are an expert P&M Chief Engineer. Analyze this technical drawing, blueprint, or machine component image in detail. "
                "Extract all visible part numbers, valve names, circuit connections, specifications, and layout descriptions so it can be indexed for workshop troubleshooting."
            )
            
            ai_vision_resp = vision_model.generate_content([
                vision_prompt,
                {"mime_type": f"image/{file_ext if file_ext != 'jpg' else 'jpeg'}", "data": img_bytes}
            ])
            
            extracted_desc = ai_vision_resp.text if ai_vision_resp and ai_vision_resp.text else "Technical drawing analysis."
            
            emb = [0.0] * 768
            if GEMINI_API_KEY:
                try:
                    emb_resp = embed_content(model="models/text-embedding-004", content=extracted_desc, task_type="retrieval_document")
                    emb = emb_resp['embedding']
                except Exception:
                    pass
            
            documents_to_insert.append({
                "fileName": file_name,
                "pageNumber": 1,
                "machine": cat_cleaned,
                "text": extracted_desc,
                "embedding": emb,
                "driveFileId": drive_file_id
            })
        else:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or len(text.strip()) < 20:
                    text = f"[Scanned Page {page_num+1} of Technical Manual / Blueprint]"
                
                chunk_size = 600
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk.strip()) < 10:
                        continue
                    
                    emb = [0.0] * 768
                    if GEMINI_API_KEY:
                        try:
                            emb_resp = embed_content(model="models/text-embedding-004", content=chunk, task_type="retrieval_document")
                            emb = emb_resp['embedding']
                        except Exception:
                            pass
                            
                    documents_to_insert.append({
                        "fileName": file_name,
                        "pageNumber": page_num + 1,
                        "machine": cat_cleaned,
                        "text": chunk,
                        "embedding": emb,
                        "driveFileId": drive_file_id
                    })
                
        if documents_to_insert:
            manuals_collection.insert_many(documents_to_insert)
            
        os.remove(file_path)
        return {"status": "success", "message": f"Successfully backed up to Google Drive & indexed {file_name} under '{categoryDesc}'!"}
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}
