# server/langchain_experiments/00_sanity_check.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # pulls from server/.env if run from langchain_experiments/, adjust path if needed

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

response = llm.invoke("Say hello in exactly 5 words.")
print(response.content)