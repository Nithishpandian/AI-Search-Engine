import os
from openai import OpenAI

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    # "gemini": {
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    #     "api_key_env": "GEMINI_API_KEY",
    #     "default_model": "gemini-3.6-flash",
    # },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
}

DEFAULT_PROVIDER = "openrouter"

_client_cache: dict[str, OpenAI] = {}

def get_client(provider: str = DEFAULT_PROVIDER) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    if provider not in _client_cache:
        config = PROVIDERS[provider]
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing environment variable: {config['api_key_env']}")

        _client_cache[provider] = OpenAI(api_key=api_key, base_url=config["base_url"])

    return _client_cache[provider]

def get_default_model(provider: str = DEFAULT_PROVIDER) -> str:
    return PROVIDERS[provider]["default_model"]