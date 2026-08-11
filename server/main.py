import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import BadRequestError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from db import Base, engine
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
from providers import get_client, get_default_model, DEFAULT_PROVIDER

load_dotenv()

from rate_limit import check_rate_limit

app = FastAPI()

Base.metadata.create_all(bind=engine)

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    provider: str = DEFAULT_PROVIDER

routing_tools = [
    {
        "type": "function",
        "function": {
            "name": "decide_approach",
            "description": "Decide how to answer the user's message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["direct", "single_search", "research"],
                        "description": (
                            "direct: answerable from conversation history alone, no search needed. "
                            "single_search: needs one straightforward web search. "
                            "research: needs multiple searches and synthesis (comparisons, multi-part questions, "
                            "questions requiring information from several independent sources)."
                        ),
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Self-contained search query if mode is single_search. Empty otherwise.",
                    },
                },
                "required": ["mode", "search_query"],
            },
        },
    }
]
MAX_RESEARCH_STEPS = 5

research_tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_research",
            "description": "Call this when you have gathered enough information to fully answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Brief summary of why you have enough information now."},
                },
                "required": ["reasoning"],
            },
        },
    },
]


def decide_approach(history: list[dict], user_message: str, provider: str = DEFAULT_PROVIDER) -> dict:
    client = get_client(provider)
    model = get_default_model(provider)

    routing_messages = [
        {
            "role": "system",
            "content": (
                "You are a routing classifier, not a chat assistant. "
                "Do NOT answer the user's message. Your only job is to call the "
                "decide_approach function to classify how their message should be handled."
            )
        }
    ] + history + [{"role": "user", "content": user_message}]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=routing_messages,
            tools=routing_tools,
            tool_choice={"type": "function", "function": {"name": "decide_approach"}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        mode = args.get("mode", "single_search")
        if mode not in ("direct", "single_search", "research"):
            mode = "single_search"

        return {
            "mode": mode,
            "search_query": args.get("search_query", "") or user_message,
        }

    except BadRequestError as e:
        # Groq returns this specific shape when the model tried to answer in
        # plain text instead of calling the forced tool. That's itself a strong
        # signal the message was answerable directly, with no search needed.
        try:
            failed_generation = e.body.get("error", {}).get("failed_generation")
        except Exception:
            failed_generation = None

        if failed_generation:
            print(f"Routing model tried to answer directly ({failed_generation!r}) — treating as mode=direct")
            return {"mode": "direct", "search_query": ""}

        print(f"Routing decision failed (bad request), defaulting to single_search: {e}")
        return {"mode": "single_search", "search_query": user_message}

    except Exception as e:
        print(f"Routing decision failed, defaulting to single_search: {e}")
        return {"mode": "single_search", "search_query": user_message}

def run_research_loop(user_question: str, history: list[dict], provider: str = DEFAULT_PROVIDER):
    """
    Runs a ReAct-style loop. Yields (event_type, data) tuples for streaming,
    and returns (all_sources, all_observations) when done.
    """
    scratchpad = history + [
        {
            "role": "system",
            "content": f"""
                You are a research assistant.

                You MUST answer ONLY by calling one of the provided tools.

                When you need information:

                - call the search tool
                - never write <function=...
                - never write XML
                - never describe the tool

                When enough information is collected,
                call finish_research.

                Maximum {MAX_RESEARCH_STEPS} searches.
            """
        },
        {"role": "user", "content": user_question},
    ]

    all_sources = []
    all_observations = []
    client = get_client(provider)
    model = get_default_model(provider)

    for step in range(MAX_RESEARCH_STEPS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=scratchpad,
                tools=research_tools,
                tool_choice="auto",
                parallel_tool_calls=False
            )
        except Exception as e:
            print(e)
            break

        message = response.choices[0].message

        if not message.tool_calls:
            # Model answered directly without calling a tool — treat as done
            break

        tool_call = message.tool_calls[0]
      
        args = json.loads(tool_call.function.arguments)

        scratchpad.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tool_call.model_dump()],
        })

        if tool_call.function.name == "finish_research":
            scratchpad.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "Research concluded.",
            })
            break

        elif tool_call.function.name == "search":
            query = args.get("query", user_question)
            results = search_web(query)
            start_index = len(all_sources) + 1

            context = format_search_context(
                results,
                start_index=start_index
            )

            sources = [
                {
                    "index": start_index + i,
                    "title": r["title"],
                    "url": r["url"]
                }
                for i, r in enumerate(results)
            ]

            all_sources.extend(sources)
            all_observations.append({"query": query, "context": context})

            yield ("research_step", {"step": step + 1, "query": query})

            scratchpad.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": context,
            })

    return all_sources, all_observations

def search_web(query: str, max_results: int = 5) -> list[dict]:
    response = tavily_client.search(query=query, max_results=max_results)
    print(response["results"])
    print("\n\n\n\n\n\n\n")
    return response["results"]  # each has: title, url, content, score

def format_search_context(
    results: list[dict],
    start_index: int = 1,
    max_chars_per_result: int = 500
):
    blocks = []

    for i, r in enumerate(results, start=start_index):
        content = r["content"][:max_chars_per_result]

        blocks.append(
            f"[{i}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{content}"
        )

    return "\n\n".join(blocks)


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.removeprefix("Bearer ")
    user_id = decode_access_token(token)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user_id
    
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

@app.post("/chat")
def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)):
    check_rate_limit(user_id)
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

    routing = decide_approach(history[:-1], request.message, provider=request.provider)

    sources = []
    system_prompt = "You are a helpful assistant."

    def event_stream():
        nonlocal sources, system_prompt

        if routing["mode"] == "research":
            gen = run_research_loop(request.message, history[:-1], provider=request.provider)

            try:
                while True:
                    event_type, data = next(gen)
                    if event_type == "research_step":
                        yield f"event: research_step\ndata: {json.dumps(data)}\n\n"
            except StopIteration as stop:
                sources, observations = stop.value
                combined_context = "\n\n".join(o["context"] for o in observations)
                system_prompt = (
                    "You are a research assistant. Using ONLY the research findings below, "
                    "give a clear, well-cited answer using [1], [2], etc. "
                    "Keep your answer focused and concise — a few short paragraphs or one summary table maximum, "
                    "not an exhaustive report.\n\n"
                    f"Research findings:\n{combined_context}"
                )

        elif routing["mode"] == "single_search":
            results = search_web(routing["search_query"])
            context = format_search_context(results)
            sources = [{"index": i, "title": r["title"], "url": r["url"]} for i, r in enumerate(results, start=1)]
            system_prompt = (
                "Answer using ONLY the search results below, citing [1], [2], etc.\n\n"
                f"Search results:\n{context}"
            )

        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        messages_for_model = [{"role": "system", "content": system_prompt}] + history

        full_reply = ""
        client = get_client(request.provider)
        model = get_default_model(request.provider)
        stream = client.chat.completions.create(
            model=model,
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