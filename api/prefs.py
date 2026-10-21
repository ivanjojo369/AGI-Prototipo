# api/prefs.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import secrets, os
import redis.asyncio as redis

app = FastAPI()
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class Pref(BaseModel):
    country: str | None  # 'es','mx','us', etc.

@app.post("/api/prefs/country")
async def set_country(pref: Pref, request: Request):
    sid = request.cookies.get("sid") or secrets.token_hex(16)
    await r.hset(f"sess:{sid}", mapping={"country": pref.country or ""})
    resp = JSONResponse({"ok": True, "country": pref.country})
    resp.set_cookie("sid", sid, httponly=True, secure=True, samesite="Lax")
    return resp

async def get_country(request: Request) -> str | None:
    sid = request.cookies.get("sid")
    if not sid: return None
    c = await r.hget(f"sess:{sid}", "country")
    return c.decode() if c else None
