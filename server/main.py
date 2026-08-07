import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from db import Base, engine
import models
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from db import get_db
from models import User, Conversation, Message
from auth import hash_password, verify_password, create_access_token
from fastapi import Depends, HTTPException, Header
from auth import decode_access_token
import os
from tavily import TavilyClient
import json

load_dotenv()

app = FastAPI()

Base.metadata.create_all(bind=engine)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory store: { conversation_id: [ {role, content}, ... ] }
conversations: dict[str, list[dict]] = {}

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

routing_tools = [
    {
        "type": "function",
        "function": {
            "name": "decide_search",
            "description": "Decide whether a web search is needed to answer the user's message, and if so, what to search for.",
            "parameters": {
                "type": "object",
                "properties": {
                  "should_search": {
                        "type": "boolean",
                        "description": "Must be a JSON boolean literal (true or false, not a string). True if answering requires current/external information not already in the conversation.",
                    },
                    "search_query": {
                        "type": "string",
                        "description": "A standalone search query, rewritten to be self-contained (resolve pronouns/context from the conversation). Empty string if should_search is false.",
                    },
                },
                "required": ["should_search", "search_query"],
            },
        },
    }
]


def search_web(query: str, max_results: int = 5) -> list[dict]:
    response = tavily_client.search(query=query, max_results=max_results)
    print(response["results"])
    print("\n\n\n\n\n\n\n")
    return response["results"]  # each has: title, url, content, score

def format_search_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}")
    print("\n\n".join(blocks))
    return "\n\n".join(blocks)


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.removeprefix("Bearer ")
    user_id = decode_access_token(token)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user_id

def decide_search(history: list[dict], user_message: str) -> dict:
    routing_messages = history + [{"role": "user", "content": user_message}]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=routing_messages,
            tools=routing_tools,
            tool_choice={"type": "function", "function": {"name": "decide_search"}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Defensive normalization: model sometimes returns "true"/"false" as strings
        should_search = args.get("should_search")
        if isinstance(should_search, str):
            should_search = should_search.strip().lower() == "true"

        return {
            "should_search": bool(should_search),
            "search_query": args.get("search_query", "") or user_message,
        }

    except Exception as e:
        print(f"Routing decision failed, defaulting to search: {e}")
        # Fail-safe: if we can't determine intent reliably, default to searching.
        # Rationale: a redundant search is wasteful; a missed one gives a stale/wrong answer.
        return {"should_search": True, "search_query": user_message}
    
@app.post("/auth/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=request.email, hashed_password=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user.id))
    return {"access_token": token, "token_type": "bearer"}

# @app.post("/chat")
# def chat(
#     request: ChatRequest,
#     user_id: str = Depends(get_current_user_id),
# ):
#     conversation_id = request.conversation_id or str(uuid.uuid4())

#     key = f"{user_id}:{conversation_id}"
#     history = conversations.get(key, [])

#     search_results = search_web(request.message)
#     context = format_search_context(search_results)

#     sources = [
#         {"index": i, "title": r["title"], "url": r["url"]}
#         for i, r in enumerate(search_results, start=1)
#     ]

#     system_prompt = (
#         "You are a helpful search assistant. Answer the user's question "
#         "using ONLY the information in the search results below. "
#         "If the results don't contain enough information to answer, say so — "
#         "do not use outside knowledge.\n\n"
#         f"Search results:\n{context}"
#     )
#     messages_for_model = [{"role": "system", "content": system_prompt}] + history
#     messages_for_model.append({"role": "user", "content": request.message})
#     history.append({"role": "user", "content": request.message})

#     print("\n\n\n\n")
#     print(conversations)

#     def event_stream():
#         yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

#         full_reply = ""
#         stream = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=messages_for_model,
#             stream=True,
#         )
#         for chunk in stream:
#             delta = chunk.choices[0].delta.content
#             if delta:
#                 full_reply += delta
#                 yield f"data: {delta}\n\n"

#         history.append({"role": "assistant", "content": full_reply})
#         conversations[key] = history
#         yield f"event: done\ndata: {conversation_id}\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")
@app.post("/chat")
def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == user_id,
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    user_msg = Message(conversation_id=conversation.id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    history = [{"role": m.role, "content": m.content} for m in conversation.messages]

    routing = decide_search(history[:-1], request.message)  # exclude the message we just added, pass it separately

    sources = []
    system_prompt = "You are a helpful assistant."

    if routing["should_search"]:
        search_results = search_web(routing["search_query"])
        context = format_search_context(search_results)
        sources = [{"index": i, "title": r["title"], "url": r["url"]} for i, r in enumerate(search_results, start=1)]
        system_prompt = (
            "You are a helpful search assistant. Answer the user's question "
            "using ONLY the information in the search results below. "
            "Cite sources inline using [1], [2], etc. matching the numbering below. "
            "If the results don't contain enough information to answer, say so.\n\n"
            f"Search results:\n{context}"
        )

    messages_for_model = [{"role": "system", "content": system_prompt}] + history

    def event_stream():
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        full_reply = ""
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                yield f"data: {delta}\n\n"

        assistant_msg = Message(conversation_id=conversation.id, role="assistant", content=full_reply)
        db.add(assistant_msg)
        db.commit()
        yield f"event: done\ndata: {conversation.id}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")