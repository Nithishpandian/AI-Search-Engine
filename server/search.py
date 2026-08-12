import json
import hashlib
from logger import logger
from tavily import TavilyClient
from rate_limit import redis_client  # reuse the same Redis connection
import os
import time

SEARCH_CACHE_TTL_SECONDS = 600  # 10 minutes
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def _cache_key(query: str) -> str:
    normalized = query.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"search_cache:{digest}"

# search.py

def search_web(query: str, max_results: int = 5) -> list[dict]:
    key = _cache_key(query)

    start = time.time()
    cached = redis_client.get(key)
    
    if cached:
        logger.info(f"Cache hit | query={query!r} | latency={time.time()-start:.3f}s")
        return json.loads(cached)

    logger.info(f"Cache miss | query={query!r}")

    response = tavily_client.search(query=query, max_results=max_results)
    results = response["results"]

    redis_client.setex(key, SEARCH_CACHE_TTL_SECONDS, json.dumps(results))

    logger.info(f"Tavily call completed | query={query!r} | latency={time.time()-start:.3f}s")

    return results