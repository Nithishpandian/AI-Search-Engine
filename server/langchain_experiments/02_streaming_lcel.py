# server/langchain_experiments/02_streaming_lcel.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# Same chain object as before. Only the method call changes.
for chunk in chain.stream({"question": "Explain what SSE is, in 3 short sentences."}):
    print(chunk, end="", flush=True)

print()  # newline after stream finishes