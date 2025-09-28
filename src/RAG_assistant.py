# src/RAG_assistant.py
from __future__ import annotations
import os, time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Iterator

from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import APIStatusError

load_dotenv()

DEFAULT_BASE_URL = os.getenv("CIRRASCALE_BASE_URL", "https://aisuite.cirrascale.com/apis/v2")
DEFAULT_API_KEY  = os.getenv("CIRRASCALE_API_KEY", "")
DEFAULT_MODEL    = os.getenv("CIRRASCALE_MODEL", "llama-3.1-8b-instruct")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Use the provided context faithfully. "
    "If the answer is not in context, say you don't know."
)

def _build_messages(system_instructions: str, context_text: str, user_text: str) -> List[Dict[str, Any]]:
    seed_context = (
        "Context below (verbatim). Use it to answer the user. If insufficient, say you don't know.\n"
        "----- BEGIN CONTEXT -----\n"
        f"{context_text}\n"
        "----- END CONTEXT -----"
    )
    return [
        {"role": "system", "content": system_instructions},
        {"role": "assistant", "content": seed_context},
        {"role": "user", "content": user_text},
    ]

def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def _candidate_base_urls(root: str) -> List[str]:
    root = root.rstrip("/")
    parts = [root, f"{root}/openai/v1", f"{root}/v1", f"{root}/openai"]
    if root.endswith("/apis/v2"):
        parts += [f"{root}/openai", f"{root}/openai/v1"]
    if "aisuite.cirrascale.com" in root:
        host_root = "https://aisuite.cirrascale.com"
        parts += [
            f"{host_root}/openai/v1",
            f"{host_root}/v1",
            f"{host_root}/apis/v2",
            f"{host_root}/apis/v2/openai",
            f"{host_root}/apis/v2/openai/v1",
        ]
    return _unique(parts)

def _probe_base_url(api_key: str, candidate: str) -> str | None:
    client = OpenAI(base_url=candidate, api_key=api_key)
    try:
        _ = client.models.list()
        return candidate
    except APIStatusError as e:
        if e.status_code in (401, 403):
            # Auth issue but path is the right API
            return candidate
        text = ""
        try: text = e.response.text or ""
        except Exception: pass
        if "<!DOCTYPE html>" in text or e.status_code == 404:
            return None
        return candidate
    except Exception:
        return None

@dataclass
class TextContextClient:
    model_id: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS

    _context_text: Optional[str] = None
    _lock: threading.Lock = threading.Lock()
    _client: Optional[OpenAI] = None

    def __post_init__(self):
        if not self.api_key:
            raise RuntimeError("Missing API key. Set CIRRASCALE_API_KEY in .env or environment.")

        # autodetect workable base url
        candidates = _candidate_base_urls(self.base_url)
        chosen = None
        for c in candidates:
            ok = _probe_base_url(self.api_key, c)
            if ok:
                chosen = ok; break
        if chosen is None:
            raise RuntimeError(
                "Could not find a working OpenAI-compatible endpoint.\n"
                "Tried:\n  - " + "\n  - ".join(candidates)
            )

        self.base_url = chosen
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Optional: validate model exists (ignore errors silently if list blocked)
        try:
            names = {m.id for m in self._client.models.list().data}
            if self.model_id not in names:
                # Don’t hard fail—just warn with a helpful message
                print(f"[RAG_assistant] Warning: model '{self.model_id}' not in provider list.\n"
                      f"Available: {sorted(list(names))[:10]}{' ...' if len(names)>10 else ''}")
        except Exception:
            pass

    # helper: conservative retries on 429
    def _with_retries(self, fn: Callable[[], Any], *, max_tries: int = 3) -> Any:
        delay = 1.0
        for attempt in range(1, max_tries + 1):
            try:
                return fn()
            except APIStatusError as e:
                if e.status_code == 429:
                    if attempt == max_tries:
                        raise RuntimeError(
                            "Rate limited by AISuite (429). Your key may lack rate limits for this model, "
                            "or the model id is not configured. Verify model name and quota in your console."
                        ) from e
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    def set_context(self, text: str) -> None:
        if text is None: text = ""
        if MAX_CONTEXT_CHARS and len(text) > MAX_CONTEXT_CHARS:
            text = text[:MAX_CONTEXT_CHARS]
        with self._lock:
            self._context_text = text

    # ---------- Streaming only ----------
    def stream(
        self,
        user_text: str,
        context_text: Optional[str] = None,
        temperature: float = 0.2,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        if context_text is not None:
            self.set_context(context_text)

        with self._lock:
            ctx = self._context_text or ""

        messages = _build_messages(self.system_instructions, ctx, user_text)
        if extra_messages:
            messages = extra_messages + messages

        # issue call with retries on 429
        def _start_stream():
            return self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
        stream = self._with_retries(_start_stream)

        for chunk in stream:
            text_piece = ""
            try:
                text_piece = chunk.choices[0].delta.get("content") or ""
            except Exception:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and hasattr(delta, "content"):
                    text_piece = delta.content or ""
                elif hasattr(chunk.choices[0], "text"):
                    text_piece = getattr(chunk.choices[0], "text") or ""
            if text_piece:
                if callback:
                    callback(text_piece)
                yield text_piece

    # Optional utility to help you confirm the exact model id
    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception as e:
            raise RuntimeError(f"Unable to list models at {self.base_url}: {e}")
