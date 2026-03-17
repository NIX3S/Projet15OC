"""
Mise à jour quotidienne des codes juridiques français
Sources : LEGITEXT + Jurisprudence Cour de cassation (travail only)
FAISS IndexFlatL2(384) + SQLite par code
"""

import os
import json
import logging
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import os

# Charge .env UNIQUEMENT s'il existe
if os.path.exists('.env'):
    load_dotenv('.env')
    print(" .env chargé")
else:
    print("ℹPas de .env trouvé")


# ── Logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
from pathlib import Path
import os

BASE_DIR = Path.cwd()

raw = os.getenv("RAG_DATA_DIR", "data")

DATA_DIR = Path(raw)

# si chemin relatif l'attacher au projet
if not DATA_DIR.is_absolute():
    DATA_DIR = BASE_DIR / DATA_DIR

DATA_DIR = DATA_DIR.resolve()
#DATA_DIR.mkdir(parents=True, exist_ok=True)

print(" DATA_DIR =", DATA_DIR)

MODEL_NAME = "dangvantuan/sentence-camembert-base"
EMBED_DIM = 768
DELTA_ALERT_THRESHOLD = 50

CODES_URLS = {
    "travail":     "LEGITEXT000006072050",
    "securite":    "LEGITEXT000006073189",
    "civil":       "LEGITEXT000006070721",
    "proc_civile": "LEGITEXT000006070716",
}

LEGIFRANCE_API_BASE = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"

# ── Helpers ─────────────────────────────────────────────

def _get_legifrance_token() -> str:
    client_id = os.getenv("LEGIFRANCE_CLIENT_ID", "")
    client_secret = os.getenv("LEGIFRANCE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise RuntimeError("Credentials Légifrance manquants (env vars LEGIFRANCE_CLIENT_ID/SECRET)")

    resp = requests.post(
        "https://sandbox-oauth.piste.gouv.fr/api/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "openid",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def _fetch_articles_for_code(code_key: str, legitext_id: str, token: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Étape 1 : Trouver les IDs avec /search
    payload_search = {
        "fond": "CODE_DATE",
        "recherche": {
            "sort": "DATE_DESC",
            "pageSize": 50,  # Petites pages pour test
            "pageNumber": 1,
            "operateur": "ET"
        }
    }
    
    r = requests.post(f"{LEGIFRANCE_API_BASE}/search", headers=headers, json=payload_search, timeout=30)
    if r.status_code != 200:
        log.error("KO %s search: %s", code_key, r.text[:300])
        return pd.DataFrame()
    
    data = r.json()
    results = data.get("results", [])
    log.info(" %s: %d résultats search", code_key, len(results))
    
    # Filtre CID + récupère les IDs
    article_ids = []
    for item in results:
        titles = item.get("titles", [])
        if any(title.get("cid") == legitext_id for title in titles):
            article_id = item.get("id")
            if article_id:
                article_ids.append(article_id)
    
    log.info(" %s: %d IDs trouvés pour CID=%s", code_key, len(article_ids), legitext_id)
    
    if not article_ids:
        return pd.DataFrame()
    
    # Étape 2 : Récupérer le texte avec /consult/getArticle
    rows = []
    for article_id in article_ids[:10]:  # Limite 10 pour test
        payload_article = {
            "id": article_id,
            "date": "2026-03-17"  # Date du jour
        }
        
        r_article = requests.post(
            f"{LEGIFRANCE_API_BASE}/consult/getArticle",
            headers=headers, 
            json=payload_article,
            timeout=30
        )
        
        if r_article.status_code == 200:
            article_data = r_article.json()
            texte = article_data.get("text", "")
            if len(texte or "") > 30:
                rows.append({
                    "id": article_id,
                    "texte": texte[:4000],
                    "statut": "VIGUEUR",
                    "date_vigueur": article_data.get("dateSignature", ""),
                    "nor": article_data.get("nor", ""),
                    "metadata": json.dumps({"code": code_key, "source": "getArticle"}),
                    "code": code_key
                })
    
    df = pd.DataFrame(rows)
    log.info("OK %s: %d articles avec texte complet", code_key, len(df))
    return df









def _fetch_jurisprudence_travail(token: str) -> pd.DataFrame:
    """Jurisprudence - Même parsing."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    payload = {
        "fond": "JURI",
        "recherche": {
            "filtres": [],
            "sort": "DATE_DESC",
            "fromAdvancedRecherche": False,
            "secondSort": "ID",
            "champs": [
                {
                    "criteres": [{"typeRecherche": "CONTIENT", "valeur": "Cour de cassation"}],
                    "operateur": "ET",
                    "typeChamp": "JURIDICTION"
                },
                {
                    "criteres": [{"typeRecherche": "CONTIENT", "valeur": "travail"}],
                    "operateur": "ET", 
                    "typeChamp": "TEXTE"
                }
            ],
            "pageSize": 100,
            "operateur": "ET",
            "typePagination": "DEFAUT",
            "pageNumber": 1
        }
    }
    
    r = requests.post(f"{LEGIFRANCE_API_BASE}/search", headers=headers, json=payload)
    if r.status_code != 200:
        return pd.DataFrame()
    
    data = r.json()
    results = data.get("results", [])
    
    rows = []
    for item in results:
        texte = item.get("text", "") or item.get("resumePrincipal", [""])[0]
        if len(texte.strip()) < 100:
            continue
            
        rows.append({
            "id": item.get("id", ""),
            "texte": texte[:4000],
            "statut": "VIGUEUR",
            "date_vigueur": item.get("dateDecision", item.get("dateSignature", "")),
            "nor": item.get("nor", ""),
            "metadata": json.dumps({"code": "jurisprudence", "juridiction": "Cour_cassation"}),
            "code": "jurisprudence",
        })
    
    return pd.DataFrame(rows)



def task_validate_indexes():
    """Valide tous les index FAISS."""
    for code_key in list(CODES_URLS.keys()) + ["jurisprudence"]:
        faiss_path = DATA_DIR / f"{code_key}.faiss"
        meta_path = DATA_DIR / f"{code_key}_meta.pkl"
        if faiss_path.exists() and meta_path.exists():
            log.info("OK %s: FAISS OK (%d vecteurs)", code_key, 
                    faiss.read_index(str(faiss_path)).ntotal)
        else:
            log.warning("KO %s: index manquant", code_key)

def task_compute_stats():
    """Statistiques globales."""
    total_articles = 0
    for code_key in list(CODES_URLS.keys()) + ["jurisprudence"]:
        df = _load_sqlite(code_key)
        vigueur = len(df[df["statut"] == "VIGUEUR"])
        total_articles += vigueur
        log.info("%s: %d articles VIGUEUR", code_key, vigueur)
    log.info("TOTAL: %d articles indexés", total_articles)

def _load_sqlite(code_key: str) -> pd.DataFrame:
    db_path = DATA_DIR / f"{code_key}.db"
    if not db_path.exists():
        return pd.DataFrame(columns=["id", "texte", "statut", "date_vigueur", "nor", "metadata", "code"])
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql("SELECT * FROM articles", conn)
        except Exception:
            return pd.DataFrame(columns=["id", "texte", "statut", "date_vigueur", "nor", "metadata", "code"])


def _save_sqlite(df: pd.DataFrame, code_key: str):
    db_path = DATA_DIR / f"{code_key}.db"
    with sqlite3.connect(db_path) as conn:
        df.to_sql("articles", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_statut ON articles(statut)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON articles(id)")
    log.info("SQLite %s → %d articles sauvegardés", code_key, len(df))


def _rebuild_faiss(df_vigueur: pd.DataFrame, code_key: str, model: SentenceTransformer):
    """Reconstruit l'index FAISS pour un code donné."""
    if df_vigueur.empty:
        log.warning("Aucun article VIGUEUR pour %s — index FAISS non créé", code_key)
        return

    texts = df_vigueur["texte"].fillna("").tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)

    faiss_path    = DATA_DIR / f"{code_key}.faiss"
    meta_path     = DATA_DIR / f"{code_key}_meta.pkl"

    faiss.write_index(index, str(faiss_path))
    meta = df_vigueur[["id", "texte", "statut", "nor", "date_vigueur", "metadata", "code"]].to_dict(orient="records")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    log.info("FAISS %s → %d vecteurs indexés", code_key, index.ntotal)


def _send_slack_alert(message: str):
    """Envoie une alerte Slack si webhook configuré."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        log.warning("SLACK_WEBHOOK_URL non configuré — alerte ignorée: %s", message)
        return
    try:
        requests.post(webhook_url, json={"text": f"🚨 RAG Juridique — {message}"}, timeout=10)
    except Exception as e:
        log.error("Slack alert failed: %s", e)

# ── Pipeline principal ──────────────────────────────────

def fetch_and_update_code(code_key: str, legitext_id: Optional[str] = None):
    log.info("=== Fetch code: %s ===", code_key)
    token = _get_legifrance_token()
    model = SentenceTransformer(MODEL_NAME)

    if code_key == "jurisprudence":
        df_delta = _fetch_jurisprudence_travail(token)
    else:
        df_delta = _fetch_articles_for_code(code_key, legitext_id, token)

    if df_delta.empty:
        log.warning("Aucune donnée reçue pour %s", code_key)
        return

    df_base = _load_sqlite(code_key)

    # Supprimer articles abrogés/remplacés
    if not df_base.empty and not df_delta.empty:
        abrogated_ids = df_delta[df_delta["statut"] == "ABROGE"]["id"].tolist()
        if abrogated_ids:
            df_base = df_base[~df_base["id"].isin(abrogated_ids)]
            log.info("%s → %d articles abrogés retirés", code_key, len(abrogated_ids))

        existing_ids = set(df_base["id"].tolist())
        df_new = df_delta[~df_delta["id"].isin(existing_ids)]
        delta_count = len(df_new)
        log.info("%s → %d nouveaux articles détectés", code_key, delta_count)

        if delta_count > DELTA_ALERT_THRESHOLD:
            _send_slack_alert(f"{delta_count} nouveaux articles détectés dans {code_key}")

        updated_ids = df_delta[df_delta["id"].isin(existing_ids)]["id"].tolist()
        if updated_ids:
            df_base = df_base[~df_base["id"].isin(updated_ids)]
        df_merged = pd.concat([df_base, df_delta[df_delta["statut"] != "ABROGE"]], ignore_index=True)
    else:
        df_merged = df_delta[df_delta["statut"] != "ABROGE"].copy()

    df_merged = df_merged.drop_duplicates(subset=["id"], keep="last")
    _save_sqlite(df_merged, code_key)
    df_vigueur = df_merged[df_merged["statut"] == "VIGUEUR"].copy()
    _rebuild_faiss(df_vigueur, code_key, model)
    log.info("=== Code %s terminé : %d articles VIGUEUR ===", code_key, len(df_vigueur))

def main():
    # 1. Mettre à jour tous les codes
    for code_key, legitext_id in CODES_URLS.items():
        fetch_and_update_code(code_key, legitext_id)

    # 2. Jurisprudence
    fetch_and_update_code("jurisprudence")

    # 3. Valider index FAISS
    task_validate_indexes()

    # 4. Statistiques
    task_compute_stats()

if __name__ == "__main__":
    main()