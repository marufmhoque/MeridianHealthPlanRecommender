"""Policy vector index (ChromaDB) used by the Policy Retriever agent.

Embeddings:
* ``OPENAI_API_KEY`` set and ``openai`` installed -> ``text-embedding-3-small``.
* Otherwise a local, deterministic hashed bag-of-words + bigram embedder, so
  the prototype runs offline, in CI, and without downloading model weights.

Policies are chunked on their markdown sections, and each numbered criterion
or bullet becomes its own chunk so retrieval returns citable units.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass

from . import config
from .data_loader import load_policy_documents
from .schemas import PolicyChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "or", "and", "to", "in", "for", "with", "at", "on", "is", "be",
    "by", "as", "than", "if", "any", "one", "following", "where", "when", "has", "been",
}


class HashingEmbedder:
    """Deterministic local embedding: signed feature hashing of unigrams+bigrams."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def name(self) -> str:  # Chroma embedding-function protocol
        return "meridian-hashing-v1"

    def _vec(self, text: str) -> list[float]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 (Chroma signature)
        return [self._vec(t) for t in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self(input)


class OpenAIEmbedder:
    def __init__(self, model: str):
        from openai import OpenAI  # optional dependency

        self.client = OpenAI()
        self.model = model

    def name(self) -> str:
        return f"openai-{self.model}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        resp = self.client.embeddings.create(model=self.model, input=input)
        return [d.embedding for d in resp.data]

    embed_documents = __call__
    embed_query = __call__


def get_embedder():
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder(config.OPENAI_EMBEDDING_MODEL)
        except Exception:
            pass
    return HashingEmbedder()


def chunk_policy(policy_id: str, markdown: str) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    section = "Header"
    title_lines: list[str] = []
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(PolicyChunk(
                chunk_id=f"{policy_id}-{len(chunks):02d}", policy_id=policy_id,
                section=section, text=body,
            ))
        buffer.clear()

    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("### "):
            flush()
            section = s[4:].rstrip(":").strip()
            # A heading that carries content ("Prior authorization required: YES") is its own chunk.
            if ":" in s and s.split(":", 1)[1].strip():
                buffer.append(s[4:])
                flush()
        elif s.startswith("#"):
            title_lines.append(s.lstrip("# "))
        elif re.match(r"^(\d+\.|-)\s", s):
            flush()
            buffer.append(s)
            flush()
        elif s:
            buffer.append(s)
    flush()
    title = " | ".join(title_lines)
    chunks.insert(0, PolicyChunk(chunk_id=f"{policy_id}-title", policy_id=policy_id, section="Title", text=title))
    return chunks


@dataclass
class PolicyIndex:
    collection: object
    chunks: dict[str, PolicyChunk]
    backend: str

    def query(self, text: str, policy_id: str | None = None, k: int = 4) -> list[PolicyChunk]:
        where = {"policy_id": policy_id} if policy_id else None
        res = self.collection.query(query_texts=[text], n_results=k, where=where)
        out: list[PolicyChunk] = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            c = self.chunks[cid].model_copy()
            c.score = round(1.0 - float(dist), 4)
            out.append(c)
        return out


_INDEX: PolicyIndex | None = None


def build_index(persist: bool = False) -> PolicyIndex:
    import chromadb

    embedder = get_embedder()
    if persist:
        config.ensure_dirs()
        client = chromadb.PersistentClient(path=str(config.STATE_DIR / "chroma"))
    else:
        client = chromadb.EphemeralClient()
    name = "meridian_policies"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name=name, embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    chunks: dict[str, PolicyChunk] = {}
    for pid, md in load_policy_documents().items():
        for c in chunk_policy(pid, md):
            chunks[c.chunk_id] = c
    col.add(
        ids=list(chunks),
        documents=[c.text for c in chunks.values()],
        metadatas=[{"policy_id": c.policy_id, "section": c.section} for c in chunks.values()],
    )
    return PolicyIndex(col, chunks, embedder.name())


def get_index() -> PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index()
    return _INDEX
