"""Swappable text encoders.

Business code calls `embed(texts, encoder, role)`. Which model sits behind it is configuration:

- `STEncoder` (default): any sentence-transformers model, run locally. Defaults to `BAAI/bge-m3`
  for resume and job retrieval (multilingual, strong Chinese/English alignment). For short
  requirement-versus-evidence matching, `intfloat/multilingual-e5-large` is the suggested model.
- `OpenAICompatibleEncoder`: a remote embeddings API, for people running this project on public data.
  It is off unless `allow_remote=True` is passed explicitly. Never use it on sensitive text.

Model families differ in how inputs must be marked (e5 expects "query: " / "passage: ", Qwen3
embeddings expect a task instruction on queries). `FAMILIES` holds those conventions, so swapping a
model never touches calling code. Embeddings are cached per model and role, so vectors from two
models are never mixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import config

DEFAULT_RETRIEVAL_MODEL = "BAAI/bge-m3"
DEFAULT_REQUIREMENT_MODEL = "intfloat/multilingual-e5-large"

TASKS = {
    "retrieval": "Given a candidate's resume, retrieve job postings the candidate is a good fit for",
    "requirement": "Given a job requirement, retrieve resume passages that show the candidate meets it",
}


@dataclass(frozen=True)
class Family:
    query_prefix: str = ""
    passage_prefix: str = ""
    instruct: bool = False              # query prefix is a task instruction


FAMILIES = {                            # matched by substring of the lower-cased model name
    "multilingual-e5": Family("query: ", "passage: "),
    "e5-": Family("query: ", "passage: "),
    "qwen3-embedding": Family(instruct=True),
    "bge-m3": Family(),
}


def family_of(model: str) -> Family:
    m = model.lower()
    return next((f for key, f in FAMILIES.items() if key in m), Family())


def prepare(texts: list[str], model: str, role: str, task: str = "retrieval") -> list[str]:
    """Apply the model family's input convention. role is 'query' or 'passage'."""
    f = family_of(model)
    if role == "query":
        prefix = f"Instruct: {TASKS[task]}\nQuery: " if f.instruct else f.query_prefix
    else:
        prefix = f.passage_prefix
    return [prefix + t for t in texts]


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], role: str = "passage", task: str = "retrieval"):
        """-> np.ndarray (n, d), L2-normalised."""
        ...


class STEncoder:
    """Local sentence-transformers model; CUDA or Apple MPS when available."""

    def __init__(self, model: str = DEFAULT_RETRIEVAL_MODEL, max_len: int = 512, batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model
        self.m = SentenceTransformer(model, device=device)
        self.m.max_seq_length = max_len
        self.batch_size = batch_size
        self.name = f"{model.replace('/', '__')}__len{max_len}"

    def encode(self, texts, role="passage", task="retrieval"):
        return self.m.encode(prepare(list(texts), self.model, role, task), batch_size=self.batch_size,
                             normalize_embeddings=True, show_progress_bar=len(texts) > 64, convert_to_numpy=True)


class OpenAICompatibleEncoder:
    """Remote embeddings over an OpenAI-compatible /embeddings endpoint. For public data only."""

    def __init__(self, model: str, base_url: str, api_key_env: str = "EMBEDDINGS_API_KEY",
                 allow_remote: bool = False, batch_size: int = 64):
        if not allow_remote:
            raise PermissionError("remote encoder disabled: pass allow_remote=True, and only for non-sensitive data")
        self.model, self.url = model, base_url.rstrip("/") + "/embeddings"
        self.key = os.environ.get(api_key_env, "")
        self.batch_size = batch_size
        self.name = f"remote__{model.replace('/', '__')}"

    def encode(self, texts, role="passage", task="retrieval"):
        import numpy as np
        out = []
        texts = prepare(list(texts), self.model, role, task)
        for i in range(0, len(texts), self.batch_size):
            req = urllib.request.Request(self.url, data=json.dumps({"model": self.model, "input": texts[i:i + self.batch_size]}).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out += [d["embedding"] for d in sorted(json.load(r)["data"], key=lambda d: d["index"])]
        v = np.asarray(out, dtype="float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def embed(texts: dict[str, str], enc: Encoder, role: str = "passage", task: str = "retrieval",
          cache_root: Path | None = None) -> dict:
    """{id: text} -> {id: vector}, cached on disk by model, role, task and text hash."""
    import numpy as np
    root = (cache_root or config.cache_dir() / "emb") / enc.name / f"{role}-{task}"
    root.mkdir(parents=True, exist_ok=True)
    key = {i: hashlib.sha256(t.encode()).hexdigest()[:32] for i, t in texts.items()}
    out, todo = {}, []
    for i, k in key.items():
        f = root / f"{k}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            todo.append(i)
    if todo:
        uniq = sorted({texts[i] for i in todo})
        vecs = dict(zip(uniq, enc.encode(uniq, role=role, task=task)))
        for i in todo:
            np.save(root / f"{key[i]}.npy", vecs[texts[i]])
            out[i] = vecs[texts[i]]
    return out
