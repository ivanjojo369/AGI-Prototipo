from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import requests
from bs4 import BeautifulSoup

# Config rápida y segura
DEFAULT_TIMEOUT = 10
RATE_LIMIT_QPS = 0.5  # 1 req cada 2s
TEXT_MAX_CHARS_DEFAULT = 80_000

_last_call_ts = 0.0

_allowed_schemes = {"http", "https"}

@dataclass
class FetchResult:
    url: str
    status: int
    title: str
    text: str


def _rate_limit():
    global _last_call_ts
    dt = time.time() - _last_call_ts
    wait = max(0.0, 1.0 / RATE_LIMIT_QPS - dt)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


def _clean_html(html: str) -> FetchResult:
    # Parser con fallback si no está lxml
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception:
        soup = BeautifulSoup(html or "", "html.parser")

    # Podamos ruido común
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    # A veces <nav>, <footer> son puro boilerplate
    for tag in soup.find_all(["nav", "footer"]):
        try:
            tag.decompose()
        except Exception:
            pass

    title = (soup.title.string or "").strip() if soup.title else ""
    text = re.sub(r"\s+", " ", soup.get_text(" ").strip())
    return FetchResult(url="", status=200, title=title, text=text)


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT, headers: Optional[Dict[str, str]] = None,
          text_max_chars: int = TEXT_MAX_CHARS_DEFAULT) -> FetchResult:
    if not url or url.split(":", 1)[0] not in _allowed_schemes:
        raise ValueError("URL no soportada")

    _rate_limit()

    ua = "AGI-Proto/1.0 (+tools.http_fetch)"
    headers = {"User-Agent": ua} | (headers or {})
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        cleaned = _clean_html(r.text)
        cleaned.url = r.url
        cleaned.status = r.status_code
        cleaned.text = cleaned.text[: max(1, int(text_max_chars or TEXT_MAX_CHARS_DEFAULT))]
        return cleaned
    except requests.exceptions.RequestException as e:
        # Error de red: devolvemos estructura homogénea
        return FetchResult(url=url, status=-1, title="", text=f"[http_fetch] error: {e}")


def run(url: str, timeout: int = DEFAULT_TIMEOUT, text_max_chars: int = TEXT_MAX_CHARS_DEFAULT, **kw) -> Dict[str, Any]:
    """
    Descarga y limpia HTML: devuelve url final, status HTTP, <title> y texto plano.
    - timeout: segundos
    - text_max_chars: tope de caracteres para 'text'
    """
    fr = fetch(url=url, timeout=timeout, text_max_chars=text_max_chars)
    return {"ok": 1 if fr.status >= 0 else 0, "url": fr.url, "status": fr.status, "title": fr.title, "text": fr.text}
