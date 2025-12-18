from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chatbot import ElectionChatbot


app = FastAPI(title="Election Chatbot", version="0.1.0")
chatbot = ElectionChatbot()
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "web"), name="static")


class ChatRequest(BaseModel):
    message: str

class AnswerRequest(BaseModel):
    question: str
    lang: str | None = None

class CategoryRequest(BaseModel):
    name: str
    lang: str | None = None

class FeedbackRequest(BaseModel):
    question: str
    helpful: bool

class RefreshRequest(BaseModel):
    secret: str | None = None

@app.post("/chat")
def chat(req: ChatRequest):
    reply = chatbot.get_response(req.message)
    return {"reply": reply}


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "web" / "index.html")


@app.get("/faqs")
def faqs():
    return {"questions": chatbot.list_questions(24)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer")
def answer(req: AnswerRequest):
    data = chatbot.get_answer_and_next_by_question(req.question, "en")
    return data


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    chatbot.record_feedback(req.question, req.helpful)
    return {"status": "ok"}

@app.post("/refresh")
def refresh(_: RefreshRequest):
    chatbot.refresh_live()
    return {"status": "ok"}


@app.get("/kb_status")
def kb_status():
    return chatbot.kb_status()
