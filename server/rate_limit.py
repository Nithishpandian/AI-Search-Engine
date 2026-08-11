import os
import time
import redis
from fastapi import HTTPException

redis_client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)

def check_rate_limit(user_id: str, limit: int = 20, window_seconds: int = 60):
    """
    Sliding-window-ish rate limit using a simple fixed window counter.
    Raises HTTPException(429) if the user has exceeded `limit` requests
    within the current `window_seconds` window.
    """
    key = f"ratelimit:{user_id}:{int(time.time()) // window_seconds}"

    current = redis_client.incr(key)
    if current == 1:
        redis_client.expire(key, window_seconds)

    if current > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {limit} requests per {window_seconds} seconds.",
        )