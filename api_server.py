# api_server.py

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from viral_meme_finder import get_recent_viral_memes

app = FastAPI(title="Viral Meme Finder API")


@app.get("/memes")
def memes(
    days_back: int = Query(14, ge=1, le=30),
    max_memes: int = Query(10, ge=1, le=50),
):
    """
    HTTP GET /memes?days_back=14&max_memes=5
    Returns JSON list of viral memes.
    """
    memes = get_recent_viral_memes(days_back=days_back, max_memes=max_memes)
    return JSONResponse(content=memes)


# Optional: simple health check
@app.get("/health")
def health():
    return {"status": "ok"}
