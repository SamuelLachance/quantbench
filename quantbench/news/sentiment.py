"""
quantbench.news.sentiment
=========================
Classement de sentiment des titres d'actualité en positif / negatif / neutre,
par LEXIQUE FINANCIER (esprit Loughran-McDonald, le standard pour le texte
financier — les mots generalistes trompent : "liability", "cut" sont negatifs en
finance). Heuristique au niveau du TITRE, calculable au build, sans API ni LLM.

Limite assumee : c'est un classifieur lexical de titres, pas de la NLP profonde.
Une amelioration possible serait FinBERT (transformer finance), plus lourd.
"""

from __future__ import annotations

import re

# Termes a polarite financiere (formes radicales ; le matching est par prefixe de mot).
_POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally",
    "rallies", "gain", "gains", "rise", "rises", "climb", "climbs", "record",
    "growth", "grow", "grows", "upgrade", "upgrades", "upgraded", "outperform",
    "outperforms", "strong", "strength", "profit", "profits", "profitable",
    "boost", "boosts", "top", "tops", "topped", "exceed", "exceeds", "beat",
    "bullish", "buyback", "buybacks", "raise", "raises", "raised", "win", "wins",
    "approval", "approved", "approve", "breakthrough", "expansion", "expand",
    "expands", "dividend", "hike", "hikes", "rebound", "rebounds", "recovery",
    "upside", "optimistic", "milestone", "partnership", "wins", "award", "awarded",
    "accelerate", "accelerates", "robust", "surged", "jumped", "soared",
    "high", "highs", "positive", "gains", "momentum", "leads", "leader",
}
_NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "drop", "drops",
    "dropped", "fall", "falls", "fell", "slump", "slumps", "decline", "declines",
    "declined", "loss", "losses", "cut", "cuts", "downgrade", "downgrades",
    "downgraded", "weak", "weakness", "warning", "warn", "warns", "warned",
    "lawsuit", "probe", "investigation", "recall", "bankruptcy", "bankrupt",
    "default", "defaults", "layoff", "layoffs", "slash", "slashes", "plummet",
    "plummets", "tumble", "tumbles", "sink", "sinks", "bearish", "fraud", "halt",
    "halts", "delay", "delays", "disappoint", "disappoints", "shortfall",
    "writedown", "writeoff", "sell-off", "selloff", "crash", "crashes", "risk",
    "risks", "concern", "concerns", "fear", "fears", "slowdown", "slowing",
    "struggle", "struggles", "low", "lows", "negative", "sued", "penalty",
    "fine", "fined", "scandal", "resign", "resigns", "cutting", "underperform",
    "downturn", "recession", "deficit", "debt", "dilution", "overvalued",
}
# Negations qui inversent la polarite du mot suivant (fenetre courte).
_NEGATORS = {"not", "no", "without", "never", "fails", "fail", "failed", "isn't",
             "doesn't", "won't", "can't", "cannot", "lack", "lacks", "avoid"}

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z'\-]*")


def classify(headline: str) -> dict:
    """Retourne {label, score, pos, neg} ; label in {positive, negative, neutral}."""
    if not headline:
        return {"label": "neutral", "score": 0, "pos": 0, "neg": 0}
    tokens = [t.lower() for t in _TOKEN.findall(headline)]
    pos = neg = 0
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in _POSITIVE else (-1 if tok in _NEGATIVE else 0)
        if polarity == 0:
            continue
        # negation dans les 2 mots precedents -> inversion
        if any(tokens[j] in _NEGATORS for j in range(max(0, i - 2), i)):
            polarity = -polarity
        if polarity > 0:
            pos += 1
        else:
            neg += 1
    score = pos - neg
    label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    return {"label": label, "score": score, "pos": pos, "neg": neg}


def fetch_and_classify(ticker: str, limit: int = 10) -> list:
    """Recupere les dernieres actus (yfinance) et les classe. Robuste au format."""
    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw[:limit]:
        c = item.get("content", item) if isinstance(item, dict) else {}
        title = c.get("title") or ""
        prov = c.get("provider") or {}
        publisher = prov.get("displayName") if isinstance(prov, dict) else (
            c.get("publisher") or "")
        url = ""
        for k in ("canonicalUrl", "clickThroughUrl"):
            v = c.get(k)
            if isinstance(v, dict) and v.get("url"):
                url = v["url"]
                break
        url = url or c.get("link", "")
        senti = classify(title)
        out.append({"title": title, "publisher": publisher, "url": url,
                    "pubDate": c.get("pubDate") or c.get("providerPublishTime"),
                    **senti})
    return out


__all__ = ["classify", "fetch_and_classify"]
