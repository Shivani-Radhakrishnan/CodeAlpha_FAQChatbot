"""Local language model (Qwen2.5-1.5B-Instruct, 4-bit GGUF) through llama.cpp.

Every generation can go through a :class:`ReplayCache` keyed by the exact prompt and
sampling settings. Evaluation runs record into it once; afterwards the published numbers
can be reproduced (in CI too) without the 1 GB model or llama.cpp installed.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"   # pinned


class ChatModel(Protocol):
    def chat(self, messages: list[dict], max_tokens: int = 200) -> str: ...


class LlamaCppModel:
    """Greedy decoding (temperature 0, fixed seed), so answers are repeatable."""

    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 4096) -> None:
        from llama_cpp import Llama

        if model_path is None:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(REPO_ID, FILENAME, revision=REVISION)
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, seed=0, verbose=False)
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], max_tokens: int = 200) -> str:
        with self._lock:
            out = self._llm.create_chat_completion(messages=messages, temperature=0.0, top_p=1.0,
                                                   max_tokens=max_tokens, seed=0)
        return out["choices"][0]["message"]["content"].strip()


class CacheMiss(RuntimeError):
    """Replay-only mode and this prompt was never recorded."""


@dataclass
class ReplayCache:
    """Wraps a model; ``model=None`` means replay only (raises :class:`CacheMiss`)."""

    path: Path
    model: Optional[ChatModel] = None

    def __post_init__(self) -> None:
        self._data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.dirty = False

    @staticmethod
    def key(messages: list[dict], max_tokens: int) -> str:
        blob = json.dumps({"m": messages, "t": max_tokens, "model": FILENAME}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def chat(self, messages: list[dict], max_tokens: int = 200) -> str:
        k = self.key(messages, max_tokens)
        if k not in self._data:
            if self.model is None:
                raise CacheMiss(k)
            self._data[k] = self.model.chat(messages, max_tokens)
            self.dirty = True
        return self._data[k]

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=0, sort_keys=True, ensure_ascii=False) + "\n",
                             encoding="utf-8")


@lru_cache(maxsize=1)
def get_model() -> LlamaCppModel:
    return LlamaCppModel()
