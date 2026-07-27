"""
quantbench.api.app
==================
API web + service des pages statiques.

Lancer :
    uvicorn quantbench.api.app:app --reload --port 8000
puis ouvrir http://localhost:8000  (page ticker) et /screener.html (screener).

Endpoints :
    GET /api/value/{ticker}?n=8000   -> valorisation complete d'un titre (live)
    GET /api/screener                -> tableau de screening (batch pre-calcule)
    GET /api/health                  -> etat
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..service import single_ticker_payload

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"

app = FastAPI(title="QuantBench API", version="1.0",
              description="Valorisation intrinseque probabiliste (DCF Monte Carlo).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])

_cache: dict = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "cached_tickers": sorted(_cache)}


@app.get("/api/value/{ticker}")
def value(ticker: str, n: int = 8000):
    ticker = ticker.upper()
    key = (ticker, n)
    if key in _cache:
        return _cache[key]
    try:
        payload = single_ticker_payload(ticker, n=n)
    except KeyError as e:                       # ticker inconnu
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, Exception) as e:        # modele inadapte / donnees manquantes
        raise HTTPException(status_code=422, detail=str(e))
    _cache[key] = payload
    return payload


@app.get("/api/screener")
def screener():
    f = APP_DIR / "screener.json"
    if not f.exists():
        raise HTTPException(status_code=404,
                            detail="screener.json absent — lancer scripts/batch_screener.py")
    return json.loads(f.read_text(encoding="utf-8"))


# Sert les pages statiques (index.html, screener.html) a la racine.
if APP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(APP_DIR), html=True), name="static")
