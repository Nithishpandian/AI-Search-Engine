# server/langchain_experiments/03_routing_structured_output.py
import os
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# This Pydantic model replaces your hand-written routing_tools JSON schema.
class RoutingDecision(BaseModel):
    """Decide how to handle the user's message."""
    mode: Literal["direct", "single_search", "research"] = Field(
        description="direct = answerable from history alone, single_search = needs one web search, research = needs multi-step research"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Self-contained rewritten search query, required if mode is single_search"
    )

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# This is the key call: binds the schema AND forces the model to use it.
structured_llm = llm.with_structured_output(RoutingDecision)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a routing classifier for a search assistant. Classify the user's message."),
    ("user", "{message}"),
])

chain = prompt | structured_llm

# Test with something that should need a search
# result = chain.invoke({"message": "What is my name?"})
# print(result)
result = chain.invoke({"message": "hmm"})
print(result)
print(type(result))
print("mode:", result.mode)