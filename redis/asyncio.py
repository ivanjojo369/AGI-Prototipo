# ejemplo de uso en tu RAG fetcher
from fastapi import APIRouter, Request
from api.prefs import get_country
from services.vpn_manager import geoclient

router = APIRouter()

@router.get("/api/websearch")
async def websearch(q: str, request: Request):
    country = await get_country(request)
    async with geoclient(country) as client:
        res = await client.get("https://www.example.com/search", params={"q": q})
        res.raise_for_status()
        return {"country": country, "html": res.text[:1500]}
