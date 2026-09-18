# server/langchain_experiments/01_basic_chat_lcel.py
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

output_parser = StrOutputParser()

# This is LCEL: the pipe chains three Runnables into one
chain = prompt | llm | output_parser

response = chain.invoke({"question": "What is FastAPI in one sentence?"})
print(response)