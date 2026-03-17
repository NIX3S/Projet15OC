"""
FastAPI RAG Service — Droit français
POST /rag/search  → chunks FAISS + metadata
POST /rag/chat    → RAG + context window + Mistral Nemo
GET  /health      → status + nb_articles_vigueur
"""

import json
import logging
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import List, Optional
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(os.getenv("RAG_DATA_DIR", Path.cwd() / "data")).resolve()
print(DATA_DIR)
MODEL_NAME  = "dangvantuan/sentence-camembert-base"
MISTRAL_URL = os.getenv("MISTRAL_API_URL", "http://localhost:11434/api/generate")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "qwen3:1.7b")
MAX_HISTORY = 10  # messages conservés dans la context window
#MISTRAL_MODEL = "mistral-nemo:latest"  # 2B → 1-2s vs 30s mistral-nemo

ALL_CODES   = ["travail", "securite", "civil", "proc_civile", "jurisprudence"]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DREETS-IA RAG Juridique",
    description="RAG offline — Codes français 2026 + Jurisprudence Cour de cassation",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modèles Pydantic ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str           # "user" | "assistant"
    content: str

class SearchRequest(BaseModel):
    question: str
    codes: List[str] = Field(default_factory=lambda: ["travail"])  # ← NOUVEAU
    top_k: int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    question: str
    codes: List[str] = Field(default_factory=lambda: ["travail"])  # ← NOUVEAU
    top_k: int = Field(default=5, ge=1, le=20)
    context_history: List[Message] = Field(default_factory=list)
    
class ChunkResult(BaseModel):
    code:         str
    article_id:   str
    texte:        str
    statut:       str
    date_vigueur: str
    nor:          str
    idcc:         Optional[str]
    juridiction:  Optional[str]
    score:        float

class SearchResponse(BaseModel):
    question:     str
    chunks:       List[ChunkResult]
    latency_ms:   float

class ChatResponse(BaseModel):
    question:     str
    answer:       str
    chunks:       List[ChunkResult]
    latency_ms:   float

# ── Index Manager ──────────────────────────────────────────────────────────────
class IndexManager:
    def __init__(self):
        self.model:   Optional[SentenceTransformer] = None
        self.indexes: dict[str, faiss.Index]        = {}
        self.metas:   dict[str, list]               = {}
        self.bm25_indexes: dict[str, BM25Okapi] = {}
        self.bm25_corpus: dict[str, list] = {}

    def load(self):
        log.info("Chargement du modèle d'embeddings : %s", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)
        self._load_all_indexes()

    def _load_all_indexes(self):
        for code_key in ALL_CODES:
            self._load_index(code_key)

    def _load_index(self, code_key: str):
        faiss_path = DATA_DIR / f"{code_key}.faiss"
        meta_path  = DATA_DIR / f"{code_key}_meta.pkl"

        if not faiss_path.exists():
            log.warning("Index FAISS absent pour %s — ignoré", code_key)
            return

        try:
            index = faiss.read_index(str(faiss_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            # Vérification cohérence
            if index.ntotal != len(meta):
                log.error("Désynchronisation FAISS/meta pour %s — reconstruction", code_key)
                self._rebuild_empty(code_key)
                return

            self.indexes[code_key] = index
            self.metas[code_key]   = meta
            log.info("✓ Index %s chargé : %d vecteurs", code_key, index.ntotal)
            corpus = [m.get("texte","") for m in meta]
            tokenized = [c.lower().split() for c in corpus]

            self.bm25_corpus[code_key] = corpus
            self.bm25_indexes[code_key] = BM25Okapi(tokenized)
        except Exception as e:
            log.error("Erreur chargement index %s : %s", code_key, e)

    def _rebuild_empty(self, code_key: str):
        """Crée un index vide en cas de corruption."""
        self.indexes[code_key] = faiss.IndexFlatIP(768)
        self.metas[code_key]   = []

    def reload(self, code_key: str):
        """Recharge un index spécifique (post mise à jour DAG)."""
        self._load_index(code_key)

    def search(self, question: str, codes: List[str], top_k: int) -> List[dict]:

        q_vec = self.model.encode([question], normalize_embeddings=True).astype("float32")

        results = []

        for code_key in codes:

            if code_key not in self.indexes:
                continue

            index = self.indexes[code_key]
            meta_list = self.metas[code_key]

            # FAISS
            k = min(top_k * 4, index.ntotal)
            distances, indices = index.search(q_vec, k)

            # BM25
            bm25 = self.bm25_indexes.get(code_key)
            bm_scores = []

            if bm25:
                tokens = question.lower().split()
                bm_scores = bm25.get_scores(tokens)

            for dist, idx in zip(distances[0], indices[0]):

                if idx < 0 or idx >= len(meta_list):
                    continue

                m = meta_list[idx]

                bm = 0
                if bm_scores is not None and len(bm_scores) > idx:
                    bm = bm_scores[idx]

                hybrid_score = dist * 0.7 + bm * 0.3
                parsed_meta = json.loads(m.get("metadata", "{}")) if isinstance(m.get("metadata"), str) else {}

                results.append({
                    "code": code_key,
                    "article_id": m.get("id",""),
                    "texte": m.get("texte","")[:1000],
                    "statut": m.get("statut",""),
                    "date_vigueur": m.get("date_vigueur",""),
                    "nor": m.get("nor",""),
                    "idcc": parsed_meta.get("idcc"),
                    "juridiction": parsed_meta.get("juridiction"),
                    "score": float(hybrid_score)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        def legal_rerank(results):
            def boost(r):

                score = r["score"]

                if r["statut"] == "VIGUEUR":
                    score += 0.3

                if r["code"] == "travail":
                    score += 0.1

                if r.get("juridiction") == "Cour de cassation":
                    score += 0.2

                return score

            results.sort(key=boost, reverse=True)
        legal_rerank(results)    
        return results[:top_k]


idx_manager = IndexManager()


@app.on_event("startup")
async def startup():
    idx_manager.load()


# ── Prompt IRAC ────────────────────────────────────────────────────────────────
PROMPT_IRAC_CONTEXTUEL = """<SYSTEM>
DREETS Hauts-de-France IA Droit 2026. Expert Paul Lesage (Data Scientist 59).

CONTEXTE CONVERSATION :
{historique}

CONTRAINTES ABSOLUES :
UNIQUEMENT textes en VIGUEUR au {date_ref}
ABROGE = signalement OBLIGATOIRE avec avertissement rouge
NOR + IDCC systématique si disponible
Cohérent avec l'historique de conversation
Jamais d'invention d'articles inexistants
</SYSTEM>

<CONTEXT_RAG>
{rag_chunks}
</CONTEXT_RAG>

<QUESTION>{question}</QUESTION>

<IRAC>
ISSUE : [Reformulation précise de la question + contexte conversation]
RULE : [Article(s) applicable(s) + NOR + statut + date_vigueur]
ANALYSIS : [Application concrète pour Utilisateur]
CONCLUSION : [Réponse synthétique + risques juridiques + prochaines étapes]
REFERENCES : [Liste numérotée des articles et jurisprudences cités]
</IRAC>"""


def _format_history(messages: List[Message]) -> str:
    """Formate les 10 derniers messages pour le prompt."""
    recent = messages[-MAX_HISTORY:]
    if not recent:
        return "Aucun historique — première question."
    lines = []
    for m in recent:
        prefix = "👤 Paul" if m.role == "user" else "🤖 DREETS-IA"
        lines.append(f"{prefix}: {m.content[:500]}")
    return "\n".join(lines)


def _format_rag_chunks(chunks: List[dict]) -> str:
    """Formate les chunks FAISS pour le prompt."""
    if not chunks:
        return "Aucun article pertinent trouvé."
    lines = []
    for i, c in enumerate(chunks, 1):
        if c["statut"] == "ABROGE":
            status_flag = "ARTICLE ABROGÉ"
        else:
            status_flag = "VIGUEUR"
        if c["statut"] == "ABROGE":
            lines.append("ATTENTION : article abrogé — vérifier version en vigueur.")
        lines.append(
            f"[{i}] {c['article_id']} ({c['code'].upper()}) — {status_flag}\n"
            f"    NOR: {c['nor'] or 'N/A'} | IDCC: {c.get('idcc') or 'N/A'} | Date: {c['date_vigueur']}\n"
            f"    {c['texte']}"
        )
    return "\n\n".join(lines)


import asyncio
from typing import List, Dict

# Ligne 25 : Modèle ULTRA-RAPIDE

async def _call_mistral(prompt: str, raw_chunks: List[Dict]) -> str:
    import httpx
    
    log.info(" Test Ollama → %s/%s", MISTRAL_URL, MISTRAL_MODEL)
    
    # TEST RAPIDE : tags (200ms)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(35.0, read=40.0)) as client:
            tags = await client.get("http://localhost:11434/api/tags")
            models = tags.json().get("models", [])
            log.info("Ollama: %d modèles: %s", len(models), [m["name"] for m in models[:3]])
            
            if MISTRAL_MODEL not in [m["name"] for m in models]:
                log.error("%s NON INSTALLÉ → ollama pull %s", MISTRAL_MODEL, MISTRAL_MODEL)
                return f"Modèle {MISTRAL_MODEL} absent → `ollama pull {MISTRAL_MODEL}`"
    except Exception as e:
        log.error("Ollama %s KO: %s", MISTRAL_URL, e)
        return "Ollama inaccessible"
    
    # PROMPT 400 chars MAX
    question = prompt
    short_prompt = f"Q: {question}\n\n{_format_rag_chunks(raw_chunks[:2])}\n\n**Réponse IRAC:**"
    
    log.info("Prompt %d chars → %s", len(short_prompt), MISTRAL_MODEL)
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(35.0, read=40.0)) as client:  # 12s MAX
            resp = await client.post(MISTRAL_URL, json={
                "model": MISTRAL_MODEL,
                "prompt": short_prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2048,      # ↓ 2k tokens
                }
            })
            
            log.info("Mistral HTTP %d (%d bytes)", resp.status_code, len(resp.content))
            resp.raise_for_status()
            data = resp.json()
            print(data)
            answer = data.get("response", "").strip()
            print(answer)
            #answer = ""

            #async for line in resp.aiter_lines():

             #   if not line:
             #       continue

             #   data = json.loads(line)

             #   if "response" in data:
             #       answer += data["response"]

            if not answer:
                raise ValueError("LLM réponse vide")
            log.info("%s: %d chars", MISTRAL_MODEL, len(answer))
            return answer
            #return resp.json()["response"].strip()[:800]
            
    except httpx.TimeoutException:
        log.error("TIMEOUT 12s → Modèle trop lent")
    except Exception as e:
        log.error("Mistral: %s", str(e)[:100])
    
    # FALLBACK RICHE
    top_chunk = raw_chunks[0] if raw_chunks else {}
    return f"LLM timeout → **{top_chunk.get('article_id', 'N/A')}** : {top_chunk.get('texte', '')[:500]}"




# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/rag/search", response_model=SearchResponse)
async def rag_search(req: SearchRequest):
    t0 = time.perf_counter()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        raw_chunks = idx_manager.search(req.question, req.codes, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("search q='%s' code=%s top_k=%d → %d chunks en %.1fms",
             req.question[:50], req.codes, req.top_k, len(raw_chunks), latency_ms)

    return SearchResponse(
        question=req.question,
        chunks=[ChunkResult(**c) for c in raw_chunks],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/rag/chat", response_model=ChatResponse)
async def rag_chat(req: ChatRequest):
    t0 = time.perf_counter()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    # RAG retrieval
    try:
        raw_chunks = idx_manager.search(req.question, req.codes, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Définir historique explicitement
    historique = _format_history(req.context_history)
    
    # Prompt avec TOUTES les variables
    prompt = PROMPT_IRAC_CONTEXTUEL.format(
        historique=historique,           # AJOUTÉ
        date_ref=time.strftime("%d/%m/%Y"),
        rag_chunks=_format_rag_chunks(raw_chunks),
        question=req.question,
    )

    # LLM avec chunks passés explicitement
    answer = await _call_mistral(prompt, raw_chunks)

    latency_ms = (time.perf_counter() - t0) * 1000
    return ChatResponse(
        question=req.question,
        answer=answer,
        chunks=[ChunkResult(**c) for c in raw_chunks],
        latency_ms=round(latency_ms, 2),
    )



@app.get("/health")
async def health():
    status = {"status": "ok", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "codes": {}}

    for code_key in ALL_CODES:
        db_path = DATA_DIR / f"{code_key}.db"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM articles WHERE statut='VIGUEUR'"
                    ).fetchone()
                    vigueur_count = row[0] if row else 0
            except Exception:
                vigueur_count = -1
        else:
            vigueur_count = 0

        faiss_path = DATA_DIR / f"{code_key}.faiss"
        faiss_count = 0
        if faiss_path.exists() and code_key in idx_manager.indexes:
            faiss_count = idx_manager.indexes[code_key].ntotal

        status["codes"][code_key] = {
            "sqlite_vigueur": vigueur_count,
            "faiss_vectors":  faiss_count,
            "index_loaded":   code_key in idx_manager.indexes,
        }

    return status


@app.post("/admin/reload/{code_key}")
async def reload_index(code_key: str):
    """Recharge un index FAISS après mise à jour par le DAG."""
    if code_key not in ALL_CODES:
        raise HTTPException(status_code=404, detail=f"Code inconnu: {code_key}")
    idx_manager.reload(code_key)
    ntotal = idx_manager.indexes.get(code_key, faiss.IndexFlatIP(768)).ntotal
    return {"code": code_key, "vectors": ntotal, "status": "reloaded"}


if __name__ == "__main__":
    uvicorn.run("rag_legal_api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
