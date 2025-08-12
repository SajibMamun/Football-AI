from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import os
import uuid
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
from motor.motor_asyncio import AsyncIOMotorClient
import re
import asyncpg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file.")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not found in .env file.")

app = FastAPI(title="FootballAI - RAG Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")

genai.configure(api_key=GOOGLE_API_KEY)

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.football_ai
sessions_collection = db.user_sessions

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "football_knowledge_base"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


@app.on_event("startup")
async def startup_event():
    app.state.pg_pool = await asyncpg.create_pool(DATABASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.pg_pool.close()


async def is_user_subscribed(email: str) -> bool:
    query = """
        SELECT is_subscribed
        FROM account_userauth
        WHERE email = $1
        LIMIT 1
    """
    async with app.state.pg_pool.acquire() as conn:
        row = await conn.fetchrow(query, email)
        if row:
            return row["is_subscribed"]
        return False


def chunk_text(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


class QuestionRequest(BaseModel):
    question: str
    session_id: str
    email: str


@app.get("/api/session")
async def get_session_id():
    new_session_id = str(uuid.uuid4())
    await sessions_collection.insert_one({
        "_id": new_session_id,
        "step": 0,
        "name": None,
        "age": None,
        "level": None,
        "club": None,
        "chat_history": [],
        "email": None  # email stored here once when known
    })
    return JSONResponse(content={"session_id": new_session_id})


@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    processed_files = []
    skipped_files = []
    errors = []
    total_chunks = 0

    total_files = len(files)
    print(f"[UPLOAD] Received {total_files} file(s) for processing.")

    for file_index, pdf in enumerate(files, start=1):
        print(f"\n[UPLOAD] ({file_index}/{total_files}) Processing file: {pdf.filename}")
        try:
            contents = await pdf.read()
            pdf.file.seek(0)

            reader = PdfReader(BytesIO(contents))
            pdf_text = ""
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
                print(f"[PAGE] Extracted text from page {page_num} of {pdf.filename}")

            if not pdf_text.strip():
                print(f"[SKIP] No text found in {pdf.filename}, skipping file.")
                skipped_files.append(pdf.filename)
                continue

            chunks = chunk_text(pdf_text)
            num_chunks = len(chunks)
            print(f"[CHUNK] Split {pdf.filename} into {num_chunks} chunk(s)")

            for chunk_idx, chunk in enumerate(chunks, start=1):
                uid = str(uuid.uuid4())
                try:
                    embedding = embeddings_model.embed_documents([chunk])
                    collection.add(
                        documents=[chunk],
                        embeddings=embedding,
                        metadatas=[{"source": pdf.filename}],
                        ids=[uid]
                    )
                    percent = (chunk_idx / num_chunks) * 100
                    print(f"[EMBED] {pdf.filename} chunk {chunk_idx}/{num_chunks} embedded ({percent:.1f}%)")
                except Exception as e:
                    error_msg = f"Embedding error on chunk {chunk_idx} of {pdf.filename}: {e}"
                    print(f"[ERROR] {error_msg}")
                    errors.append(error_msg)

            processed_files.append(pdf.filename)
            total_chunks += num_chunks
            print(f"[COMPLETE] Finished embedding all chunks for {pdf.filename}")

        except Exception as e:
            error_msg = f"Failed to process file {pdf.filename}: {e}"
            print(f"[ERROR] {error_msg}")
            errors.append(error_msg)

    print(f"\n[SUMMARY] Processed: {len(processed_files)} files, Skipped: {len(skipped_files)} files, Total chunks embedded: {total_chunks}")
    if errors:
        print(f"[SUMMARY] Encountered {len(errors)} errors during upload and embedding.")

    return {
        "message": f"Processed {len(processed_files)} PDFs, skipped {len(skipped_files)} PDFs.",
        "files_processed": processed_files,
        "files_skipped": skipped_files,
        "total_chunks_stored": total_chunks,
        "errors": errors
    }


async def get_session(session_id: str):
    session = await sessions_collection.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def update_session(session_id: str, data: dict):
    await sessions_collection.update_one({"_id": session_id}, {"$set": data})


async def append_chat_history(session_id: str, role: str, text: str):
    await sessions_collection.update_one(
        {"_id": session_id},
        {"$push": {"chat_history": {"role": role, "text": text}}}
    )


@app.post("/ai/chat")
async def ask_question(request: QuestionRequest):
    session_id = request.session_id.strip()
    email = request.email.strip()
    question = (request.question or "").strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    # ✅ Check subscription
    if not await is_user_subscribed(email):
        raise HTTPException(
            status_code=403,
            detail="You are not subscribed. Please subscribe to use this service."
        )

    # Get or create session
    try:
        session = await get_session(session_id)
    except HTTPException:
        # New session that wasn't created through /api/session
        session = {
            "_id": session_id,
            "step": 0,
            "name": None,
            "age": None,
            "level": None,
            "club": None,
            "chat_history": [],
            "email": email,
        }
        await sessions_collection.insert_one(session)

    # Update email if missing or changed
    if session.get("email") != email:
        await update_session(session_id, {"email": email})

    step = session.get("step", 0)

    def is_valid_name(name):
        return bool(re.match(r"^[A-Za-z ]{2,}$", name))

    def is_valid_age(age):
        return age.isdigit() and 5 <= int(age) <= 100

    def is_valid_level(level):
        return level.lower() in {"beginner", "amateur", "semi-pro", "pro"}

    def is_valid_club(club):
        return len(club) >= 2

    # ONBOARDING FLOW (steps 0 to 3)
    if step < 4:
        # If no question given yet, prompt next onboarding question
        if not question:
            if step == 0:
                bot_msg = "What is your name?"
            elif step == 1:
                bot_msg = "How old are you?"
            elif step == 2:
                bot_msg = "What is your current playing level? (Beginner, Amateur, Semi-pro, Pro)"
            elif step == 3:
                bot_msg = "Which club do you play for?"
            else:
                bot_msg = "Let's continue."

            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

        # Validate input and proceed onboarding
        if step == 0:
            if not is_valid_name(question):
                bot_msg = "Please enter a valid name (only letters and spaces, min 2 characters)."
                await append_chat_history(session_id, "bot", bot_msg)
                return {"answer": bot_msg}

            await update_session(session_id, {"name": question, "step": 1})
            await append_chat_history(session_id, "user", question)
            bot_msg = "How old are you?"
            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

        elif step == 1:
            if not is_valid_age(question):
                bot_msg = "Please enter a valid age (a number between 5 and 100). How old are you?"
                await append_chat_history(session_id, "bot", bot_msg)
                return {"answer": bot_msg}

            await update_session(session_id, {"age": question, "step": 2})
            await append_chat_history(session_id, "user", question)
            bot_msg = "What is your current playing level? (Beginner, Amateur, Semi-pro, Pro)"
            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

        elif step == 2:
            if not is_valid_level(question):
                bot_msg = "Please enter a valid playing level: Beginner, Amateur, Semi-pro, or Pro."
                await append_chat_history(session_id, "bot", bot_msg)
                return {"answer": bot_msg}

            await update_session(session_id, {"level": question.title(), "step": 3})
            await append_chat_history(session_id, "user", question)
            bot_msg = "Which club do you play for?"
            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

        elif step == 3:
            if not is_valid_club(question):
                bot_msg = "Please enter a valid club name (at least 2 characters). Which club do you play for?"
                await append_chat_history(session_id, "bot", bot_msg)
                return {"answer": bot_msg}

            await update_session(session_id, {"club": question, "step": 4})
            await append_chat_history(session_id, "user", question)
            session = await get_session(session_id)  # Refresh session for name
            bot_msg = f"Great! Nice to meet you {session.get('name', 'there')}! How can I help you become a better player?"
            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

    # AFTER ONBOARDING (step >= 4), handle football questions
    if step >= 4:
        if not question:
            bot_msg = "Please ask a football-related question."
            await append_chat_history(session_id, "bot", bot_msg)
            return {"answer": bot_msg}

        try:
            await append_chat_history(session_id, "user", question)
            session = await get_session(session_id)

            query_embedding = embeddings_model.embed_query(question)
            results = collection.query(query_embeddings=[query_embedding], n_results=4)
            documents = results.get("documents", [[]])[0]
            context = "\n\n".join(documents).strip()

            prompt = f"""
You are a specialized football expert AI assistant with access to a curated knowledge base.

STRICT INSTRUCTIONS:
1. ONLY use information from the provided knowledge base - never add external knowledge
2. Give specific answer.
3. Tailor your response specifically to the user's profile.
4. Be precise, specific, and concise in your answers

USER PROFILE:
- Name: {session.get('name', 'Unknown')}
- Age: {session.get('age', 'Unknown')}
- Skill Level: {session.get('level', 'Unknown')}
- Club: {session.get('club', 'Unknown')}

RESPONSE GUIDELINES:
- AGE-APPROPRIATE: Answer might be user age-specific.
- LEVEL-SPECIFIC: Tailor technical depth to skill level (beginner/intermediate/advanced)
- CLUB-RELEVANT: When applicable, reference their club context for personalized advice
- PRECISION: Provide specific, actionable information rather than generic responses
- BREVITY: Keep answers concise while maintaining completeness

KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

ANSWER REQUIREMENTS:
- Start directly with the answer (no "Based on the knowledge base..." preambles)
- Use age-appropriate language and examples
- Match technical complexity to skill level
- Include club-specific considerations when relevant
- Cite specific details from the knowledge base
- If information is partial, state what's available and what's missing
- Maximum 150 words unless question requires detailed explanation
"""

            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)

            bot_msg = response.text.strip()
            await append_chat_history(session_id, "bot", bot_msg)

            return {"answer": bot_msg}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")



@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/chat.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading UI: {e}</h1>"
