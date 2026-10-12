"""Sentence embeddings with all-MiniLM-L6-v2, run through ONNX Runtime (no PyTorch).

The same model the sentence-transformers library would load, exported to ONNX by its
authors; about 90 MB, downloaded from Hugging Face on first use and cached, or read from
``BOTBUDDY_MODEL_DIR``. Using ONNX Runtime keeps PyTorch (several hundred MB) out of the
install.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned, so embeddings are reproducible
MAX_TOKENS = 128


def model_files() -> tuple[Path, Path]:
    local = os.environ.get("BOTBUDDY_MODEL_DIR")
    if local:
        return Path(local) / "model.onnx", Path(local) / "tokenizer.json"
    from huggingface_hub import hf_hub_download

    return (Path(hf_hub_download(REPO_ID, "onnx/model.onnx", revision=REVISION)),
            Path(hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION)))


class Encoder:
    """Thread-safe: ONNX Runtime sessions can be shared; the tokenizer is guarded."""

    dim = 384

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path, tok_path = model_files()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Unit-length embeddings, shape (len(texts), 384)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            with self._lock:
                enc = self.tokenizer.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]                      # (batch, tokens, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    return Encoder()
