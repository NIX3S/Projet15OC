"""
Bootstrap initial : XML Légifrance → SQLite → FAISS
Télécharge, parse et indexe les 4 codes + jurisprudence
Usage : python data/bootstrap.py [--codes travail securite civil proc_civile] [--test]
"""

import argparse
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import httpx
import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bootstrap.log")],
)

# ── Dépendances lourdes (import conditionnel pour tests) ───────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_ML = True
except ImportError:
    HAS_ML = False
    log.warning("faiss ou sentence-transformers non installés — étape FAISS désactivée")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(os.getenv("RAG_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "dangvantuan/sentence-camembert-base"
EMBED_DIM   = 768
BATCH_SIZE  = 64

CODES_URLS = {
    "travail":     "LEGITEXT000006072050",
    "securite":    "LEGITEXT000006073189",
    "civil":       "LEGITEXT000006070721",
    "proc_civile": "LEGITEXT000006070716",
}

# URLs de téléchargement XML Légifrance (format Legi)
XML_URLS = {
    "travail":     "https://codes.droit.org/payloads/Code%20du%20travail.xml",
    "securite":    "https://codes.droit.org/payloads/Code%20de%20la%20s%C3%A9curit%C3%A9%20sociale.xml",
    "civil":       "https://codes.droit.org/payloads/Code%20civil.xml",
    "proc_civile": "https://codes.droit.org/payloads/Code%20de%20proc%C3%A9dure%20civile.xml",
}

# ------------ Parsers XML Légifrance ------------------------------------------

def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    # Supprimer balises HTML résiduelles
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_statut(etat: str) -> str:
    mapping = {
        "VIGUEUR": "VIGUEUR",
        "VIGUEUR_DIFF": "VIGUEUR",
        "ABROGE": "ABROGE",
        "ABROGE_DIFF": "ABROGE",
        "MODIFIE": "MODIFIE",
        "TRANSFERE": "ABROGE",
        "PERIME": "ABROGE",
    }
    return mapping.get(etat.upper().strip(), "VIGUEUR")


def parse_legi_xml_file(xml_path: Path, code_key: str) -> pd.DataFrame:
    """Parse HIÉRARCHIQUE : Section(title) + Article(num) → chunk contextuel."""
    rows = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        log.warning("XML parse error %s: %s", xml_path, e)
        return pd.DataFrame()

    # Parcourir HIÉRARCHIE <t title="..."> ,  <article>
    for t in root.iter("t"):
        section_title = t.get("title", "Sans section")
        
        for article in t.iter("article"):
            article_id = article.get("id", "")
            num = article.get("num", "")
            etat = article.get("etat", "VIGUEUR")
            
            # EXTRAIRE TEXTE complet (article + enfants)
            texte_parts = [child.text.strip() for child in article.iter() 
                          if child.text and child.text.strip()]
            texte = _clean_text(" ".join(texte_parts))
            
            if len(texte) < 30:  # Minimum pertinent
                continue

            statut = _parse_statut(etat)
            
            #  CHUNK JURIDIQUE = SECTION + ARTICLE NUMÉROTÉ 
            chunk_enrichi = f"[{code_key.upper()}] {section_title}\nArt. {num} : {texte}"
            
            metadata = {
                "code": code_key,
                "section": section_title[:200],
                "article_num": num,
                "article_id": article_id,
                "statut": statut,
                "date_vigueur": article.get("date", ""),
            }

            rows.append({
                "id": article_id,
                "texte": chunk_enrichi[:1900],  # Embedding-friendly
                "statut": statut,
                "date_vigueur": metadata["date_vigueur"],
                "nor": article.get("nor", ""),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "code": code_key,
                "article_num": num,
                "section": section_title,
            })

    log.info("✓ %s : %d chunks HIÉRARCHIQUES (avec sections)", code_key, len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["id", "texte", "statut", "date_vigueur", "nor", "metadata", "code", "article_num", "section"]
    )




def parse_legi_xml_directory(xml_dir: Path, code_key: str) -> pd.DataFrame:
    """Parse XML ou PDF dans le répertoire."""
    all_dfs = []
    
    # Cherche tous les fichiers XML + PDF
    xml_files = list(xml_dir.rglob("*.xml")) + list(xml_dir.rglob("*.pdf"))
    if not xml_files:
        log.warning("Aucun fichier XML/PDF trouvé dans %s", xml_dir)
        return pd.DataFrame()
    
    log.info("Parsing %d fichiers pour %s", len(xml_files), code_key)

    for xml_file in xml_files:
        if xml_file.suffix == ".xml":
            df = parse_legi_xml_file(xml_file, code_key)
        else:  # PDF fallback minimal
            log.warning("PDF détecté %s — parsing simplifié", xml_file)
            df = pd.DataFrame()  # À implémenter si besoin
        
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["id", "texte", "statut", "date_vigueur", "nor", "metadata", "code"])

    df_all = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["id"])
    log.info("✓ %s : %d articles parsés", code_key, len(df_all))
    return df_all


# --- Téléchargement ----------------------------------------------

def download_and_extract(code_key: str, force: bool = False) -> Optional[Path]:
    """Télécharge le fichier source (XML/PDF) sans extraction forcée."""
    url = XML_URLS.get(code_key)
    if not url:
        log.warning("Pas d'URL pour %s", code_key)
        return None

    dest_dir = DATA_DIR / f"xml_{code_key}"
    if dest_dir.exists() and not force and any(dest_dir.rglob("*.xml")):
        log.info("Données XML existantes pour %s", code_key)
        return dest_dir

    # Nom de fichier sans .tar.gz trompeur
    ext = "xml" if "xml" in url.lower() else "pdf"
    source_file = DATA_DIR / f"{code_key}.{ext}"
    
    log.info("Téléchargement %s → %s", url, source_file)
    
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(source_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        log.info("Téléchargé %.1f MB", source_file.stat().st_size / 1024 / 1024)
    except requests.RequestException as e:
        log.error("Échec téléchargement %s: %s", code_key, e)
        return None

    dest_dir.mkdir(exist_ok=True)
    dest_file = dest_dir / source_file.name
    
    # Copie directe (pas d'extraction)
    source_file.rename(dest_file)
    log.info("Fichier placé → %s", dest_file)
    
    return dest_dir



# ── Via API Légifrance (fallback) ──────────────────────────────────────────────

def fetch_via_api(code_key: str, legitext_id: str) -> pd.DataFrame:
    """Fallback : récupération via API PISTE Légifrance."""
    client_id     = os.getenv("LEGIFRANCE_CLIENT_ID", "")
    client_secret = os.getenv("LEGIFRANCE_CLIENT_SECRET", "")

    if not client_id:
        log.warning("API Légifrance non configurée (LEGIFRANCE_CLIENT_ID manquant)")
        return pd.DataFrame()

    log.info("Récupération via API PISTE pour %s (%s)", code_key, legitext_id)

    # Token
    try:
        token_resp = requests.post(
            "https://oauth.piste.gouv.fr/api/oauth/token",
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
                "scope":         "openid",
            },
            timeout=30,
        )
        token_resp.raise_for_status()
        token = token_resp.json()["access_token"]
    except Exception as e:
        log.error("Échec authentification PISTE: %s", e)
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    rows    = []
    page    = 1

    while True:
        try:
            r = requests.post(
                "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/search",
                headers=headers,
                json={
                    "fond":       "CODE_DATE",
                    "recherche":  {"champs": [{"typeChamp": "ALL", "criteres": [{"typeRecherche": "EXACTE", "valeur": "*"}]}]},
                    "pageNumber": page,
                    "pageSize":   500,
                    "textId":     legitext_id,
                },
                timeout=60,
            )
            r.raise_for_status()
        except Exception as e:
            log.error("API page %d: %s", page, e)
            break

        data    = r.json()
        results = data.get("results", [])
        if not results:
            break

        for art in results:
            texte = _clean_text(art.get("texte", ""))
            if not texte:
                continue
            statut = _parse_statut(art.get("etat", "VIGUEUR"))
            metadata = {
                "code":        code_key,
                "id_nor":      art.get("nor", ""),
                "statut":      statut,
                "date_vigueur": art.get("dateDebut", ""),
                "idcc":        art.get("idcc", ""),
                "article_id":  art.get("id", ""),
                "juridiction": None,
            }
            rows.append({
                "id":          art.get("id", ""),
                "texte":       texte[:2000],
                "statut":      statut,
                "date_vigueur": art.get("dateDebut", ""),
                "nor":         art.get("nor", ""),
                "metadata":    json.dumps(metadata, ensure_ascii=False),
                "code":        code_key,
            })

        total = data.get("totalResultNumber", 0)
        if page * 500 >= total:
            break
        page += 1
        time.sleep(0.2)  # Rate limit

    log.info("API → %d articles pour %s", len(rows), code_key)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["id", "texte", "statut", "date_vigueur", "nor", "metadata", "code"]
    )


# ── SQLite ─────────────────────────────────────────────────────────────────────

def save_to_sqlite(df: pd.DataFrame, code_key: str):
    db_path = DATA_DIR / f"{code_key}.db"
    with sqlite3.connect(db_path) as conn:
        df.to_sql("articles", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_statut ON articles(statut)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON articles(id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code ON articles(code)")
    log.info("SQLite %s → %d articles (%.1f MB)", code_key, len(df),
             db_path.stat().st_size / 1024 / 1024)


# ── FAISS ───────────────────────────────────────────────────────────────────────

def build_faiss_index(df_vigueur: pd.DataFrame, code_key: str, model: "SentenceTransformer"):
    if df_vigueur.empty:
        log.warning("Aucun article VIGUEUR pour %s — FAISS vide créé", code_key)
        index = faiss.IndexFlatL2(EMBED_DIM)
        faiss.write_index(index, str(DATA_DIR / f"{code_key}.faiss"))
        with open(DATA_DIR / f"{code_key}_meta.pkl", "wb") as f:
            pickle.dump([], f)
        return

    texts = df_vigueur["texte"].fillna("").tolist()
    log.info("Calcul embeddings %s : %d textes...", code_key, len(texts))

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        emb   = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)
        if (i // BATCH_SIZE) % 10 == 0:
            log.info("  Batch %d/%d", i // BATCH_SIZE + 1, (len(texts) // BATCH_SIZE) + 1)

    embeddings = np.vstack(all_embeddings).astype("float32")

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)

    faiss_path = DATA_DIR / f"{code_key}.faiss"
    meta_path  = DATA_DIR / f"{code_key}_meta.pkl"

    faiss.write_index(index, str(faiss_path))
    meta = df_vigueur[["id", "texte", "statut", "nor", "date_vigueur", "metadata", "code"]].to_dict(orient="records")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    log.info("✓ FAISS %s : %d vecteurs (%.1f MB)", code_key, index.ntotal,
             faiss_path.stat().st_size / 1024 / 1024)


# ── Test RAG inline ────────────────────────────────────────────────────────────

def test_rag_basic(model: "SentenceTransformer"):
    """Test : 'préavis démission' → Art. L1237-1 + section contrat."""
    code_key = "travail"
    faiss_path = DATA_DIR / f"{code_key}.faiss"
    meta_path = DATA_DIR / f"{code_key}_meta.pkl"

    if not faiss_path.exists():
        log.warning("Test impossible — index absent")
        return

    index = faiss.read_index(str(faiss_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    q_vec = model.encode(["préavis démission CDI"], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(q_vec, 5)

    log.info("=== Test RAG 'préavis démission' ===")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        m = meta[idx]
        section = m.get("section", "Sans section")[:50]
        article_num = m.get("article_num", "")
        log.info(" %d. Art.%s [%s...] | score=%.3f", i+1, article_num, section, dist)
        
        # Succès = L1237 OU mots-clés + section pertinente
        if any(x in m["texte"].upper() for x in ["L1237", "PRÉAVIS", "DÉMISSION"]):
            log.info("OK HIT %d : Art. %s trouvé !", i+1, article_num)


# ── Main bootstrap ─────────────────────────────────────────────────────────────

def bootstrap_code(code_key: str, use_api: bool = False, force_download: bool = False):
    log.info("=== Bootstrap code: %s ===", code_key)
    legitext_id = CODES_URLS.get(code_key)

    df = pd.DataFrame()

    if not use_api:
        # Essayer XML d'abord
        xml_dir = DATA_DIR / f"xml_{code_key}"
        if not xml_dir.exists() or force_download:
            xml_dir = download_and_extract(code_key, force=force_download)

        if xml_dir and xml_dir.exists():
            df = parse_legi_xml_directory(xml_dir, code_key)

    if df.empty and legitext_id:
        log.info("Fallback API pour %s", code_key)
        df = fetch_via_api(code_key, legitext_id)

    if df.empty:
        log.error("Aucune donnée pour %s — bootstrap incomplet", code_key)
        return False

    # Nettoyage final
    df = df[df["texte"].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=["id"], keep="last")

    # SQLite
    save_to_sqlite(df, code_key)

    return True


def main():
    parser = argparse.ArgumentParser(description="Bootstrap RAG Juridique")
    parser.add_argument("--codes", nargs="+",
                        default=list(CODES_URLS.keys()),
                        help="Codes à indexer")
    parser.add_argument("--api",        action="store_true", help="Forcer l'API Légifrance")
    parser.add_argument("--force",      action="store_true", help="Re-télécharger les XML")
    parser.add_argument("--test",       action="store_true", help="Lancer le test RAG après bootstrap")
    parser.add_argument("--no-faiss",   action="store_true", help="Skiper l'étape FAISS")
    args = parser.parse_args()

    t_start = time.perf_counter()
    log.info("=== BOOTSTRAP INITIAL RAG JURIDIQUE ===")
    log.info("Codes : %s", args.codes)
    log.info("DATA_DIR : %s", DATA_DIR.absolute())

    success_codes = []
    for code_key in args.codes:
        ok = bootstrap_code(code_key, use_api=args.api, force_download=args.force)
        if ok:
            success_codes.append(code_key)

    if not success_codes:
        log.error("Aucun code bootstrappé avec succès — vérifier les credentials API")
        sys.exit(1)

    # FAISS
    if not args.no_faiss and HAS_ML:
        log.info("Chargement modèle embeddings : %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)

        for code_key in success_codes:
            db_path = DATA_DIR / f"{code_key}.db"
            with sqlite3.connect(db_path) as conn:
                df_vigueur = pd.read_sql(
                    "SELECT * FROM articles WHERE statut='VIGUEUR'", conn
                )
            build_faiss_index(df_vigueur, code_key, model)

        if args.test:
            test_rag_basic(model)
    elif args.no_faiss:
        log.info("Étape FAISS skippée (--no-faiss)")
    else:
        log.warning("ML libs non disponibles — étape FAISS skippée")

    elapsed = time.perf_counter() - t_start
    log.info("=== Bootstrap terminé en %.1fs — %d codes indexés ===",
             elapsed, len(success_codes))

    # Résumé
    print("\n" + "="*50)
    print("OK BOOTSTRAP TERMINÉ")
    print(f"   Durée : {elapsed:.0f}s")
    for code_key in success_codes:
        db_path    = DATA_DIR / f"{code_key}.db"
        faiss_path = DATA_DIR / f"{code_key}.faiss"
        print(f"   {code_key:15} SQLite={'OK' if db_path.exists() else 'KO'} "
              f"FAISS={'OK' if faiss_path.exists() else 'KO'}")
    print("="*50)


if __name__ == "__main__":
    main()
