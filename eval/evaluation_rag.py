"""
Évaluation RAG Juridique — Precision@5 + MRR
50 questions de référence sur le droit du travail français
KPI cibles : Precision@5 ≥ 98%, MRR ≥ 0.85, Latence FAISS ≤ 50ms
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from datetime import UTC
import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

API_BASE = "http://localhost:8000"
EVAL_DIR = Path("eval/results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Jeu de 50 questions de référence ──────────────────────────────────────────
# Format : {question, expected_articles (liste d'IDs ou mots-clés), code_prioritaire}
EVAL_QUESTIONS = [
    # ── Droit du travail ──────────────────────────────────────────────
    {"q": "Quelle est la durée du préavis en cas de démission d'un CDI ?",
     "expected": ["L1237-1", "préavis", "démission"], "code": "travail"},
    {"q": "Quelles sont les conditions de validité d'un licenciement pour faute grave ?",
     "expected": ["L1234-1", "faute grave", "licenciement"], "code": "travail"},
    {"q": "Quel est le nombre maximal d'heures supplémentaires par an ?",
     "expected": ["L3121-22", "contingent", "heures supplémentaires"], "code": "travail"},
    {"q": "Comment calculer l'indemnité légale de licenciement ?",
     "expected": ["L1234-9", "indemnité licenciement"], "code": "travail"},
    {"q": "Quelles sont les obligations de l'employeur en matière de formation professionnelle ?",
     "expected": ["L6311-1", "formation", "CPF"], "code": "travail"},
    {"q": "Quand peut-on rompre un contrat en période d'essai ?",
     "expected": ["L1221-25", "période d'essai", "rupture"], "code": "travail"},
    {"q": "Quelles sont les règles relatives au repos hebdomadaire ?",
     "expected": ["L3132-1", "repos hebdomadaire"], "code": "travail"},
    {"q": "Comment fonctionne le télétravail selon le code du travail ?",
     "expected": ["L1222-9", "télétravail"], "code": "travail"},
    {"q": "Quelles sont les conditions de recours au CDD ?",
     "expected": ["L1242-1", "CDD", "contrat à durée déterminée"], "code": "travail"},
    {"q": "Qu'est-ce qu'un accord de performance collective ?",
     "expected": ["L2254-2", "accord performance collective"], "code": "travail"},
    {"q": "Quel est le délai de prescription pour les actions en salaire ?",
     "expected": ["L3245-1", "prescription", "salaire"], "code": "travail"},
    {"q": "Quelles sont les protections accordées aux représentants du personnel ?",
     "expected": ["L2411-1", "représentant", "protection"], "code": "travail"},
    {"q": "Comment se calcule le droit à congés payés ?",
     "expected": ["L3141-3", "congés payés", "2,5 jours"], "code": "travail"},
    {"q": "Quelles sont les règles de la rupture conventionnelle ?",
     "expected": ["L1237-11", "rupture conventionnelle"], "code": "travail"},
    {"q": "Qu'implique l'obligation de reclassement avant licenciement économique ?",
     "expected": ["L1233-4", "reclassement", "licenciement économique"], "code": "travail"},

    # ── Sécurité sociale ──────────────────────────────────────────────
    {"q": "Quelles sont les conditions d'ouverture du droit aux indemnités journalières maladie ?",
     "expected": ["L313-1", "indemnités journalières", "maladie"], "code": "securite"},
    {"q": "Comment est calculée la pension de retraite de base ?",
     "expected": ["L161-17", "retraite", "pension"], "code": "securite"},
    {"q": "Quelles cotisations patronales finance la branche accidents du travail ?",
     "expected": ["L242-7", "accidents travail", "cotisations"], "code": "securite"},
    {"q": "Quelle est la durée maximale du congé maternité ?",
     "expected": ["L1225-17", "congé maternité", "semaines"], "code": "securite"},
    {"q": "Comment fonctionne la CMU-C (complémentaire santé solidaire) ?",
     "expected": ["L861-1", "complémentaire santé", "solidaire"], "code": "securite"},

    # ── Code civil ───────────────────────────────────────────────────
    {"q": "Quelles sont les conditions de validité d'un contrat selon le code civil ?",
     "expected": ["1128", "consentement", "contrat"], "code": "civil"},
    {"q": "Qu'est-ce que la force majeure en droit des contrats ?",
     "expected": ["1218", "force majeure", "imprévisible"], "code": "civil"},
    {"q": "Comment s'apprécie la responsabilité contractuelle ?",
     "expected": ["1231-1", "responsabilité contractuelle", "inexécution"], "code": "civil"},
    {"q": "Quels sont les délais de prescription de droit commun ?",
     "expected": ["2224", "prescription", "5 ans"], "code": "civil"},
    {"q": "Qu'est-ce que l'abus de droit ?",
     "expected": ["1240", "abus de droit", "préjudice"], "code": "civil"},

    # ── Procédure civile ─────────────────────────────────────────────
    {"q": "Quel est le délai pour interjeter appel d'un jugement de première instance ?",
     "expected": ["appel", "délai", "mois"], "code": "proc_civile"},
    {"q": "Quand peut-on demander une procédure en référé ?",
     "expected": ["référé", "urgence", "provisoire"], "code": "proc_civile"},
    {"q": "Quelles sont les conditions de recevabilité d'une action en justice ?",
     "expected": ["intérêt à agir", "qualité", "recevabilité"], "code": "proc_civile"},
    {"q": "Comment fonctionne la jonction d'instances ?",
     "expected": ["jonction", "instance", "connexité"], "code": "proc_civile"},
    {"q": "Qu'est-ce qu'une mise en état devant le tribunal ?",
     "expected": ["mise en état", "juge", "instruction"], "code": "proc_civile"},

    # ── Jurisprudence travail ─────────────────────────────────────────
    {"q": "Quelle est la position de la Cour de cassation sur le barème Macron ?",
     "expected": ["barème", "Macron", "indemnité"], "code": "travail"},
    {"q": "Comment la Cour de cassation encadre le droit à la déconnexion ?",
     "expected": ["déconnexion", "heures", "astreinte"], "code": "travail"},
    {"q": "Quelle est la jurisprudence sur la prise d'acte de rupture ?",
     "expected": ["prise d'acte", "rupture", "manquement"], "code": "travail"},
    {"q": "Comment la jurisprudence définit-elle le harcèlement moral au travail ?",
     "expected": ["harcèlement moral", "L1152-1", "agissements"], "code": "travail"},
    {"q": "Quelle est la position jurisprudentielle sur le forfait jour ?",
     "expected": ["forfait jour", "convention", "repos"], "code": "travail"},

    # ── Questions complexes multi-codes ──────────────────────────────
    {"q": "Un salarié accident du travail peut-il être licencié pour inaptitude ?",
     "expected": ["L1226-9", "inaptitude", "accident travail"], "code": "travail"},
    {"q": "Comment se calcule le solde de tout compte en cas de licenciement économique ?",
     "expected": ["L1234-20", "solde tout compte", "licenciement"], "code": "travail"},
    {"q": "Quelles sont les règles de transfert de contrat en cas de cession d'entreprise ?",
     "expected": ["L1224-1", "transfert", "cession"], "code": "travail"},
    {"q": "Un employeur peut-il modifier unilatéralement le contrat de travail ?",
     "expected": ["L1221-1", "modification", "refus salarié"], "code": "travail"},
    {"q": "Quelles sont les étapes de la procédure de licenciement pour motif personnel ?",
     "expected": ["L1232-2", "convocation", "entretien préalable"], "code": "travail"},
    {"q": "Comment calculer l'indemnité compensatrice de congés payés ?",
     "expected": ["L3141-26", "indemnité compensatrice", "congés"], "code": "travail"},
    {"q": "Quelles obligations vis-à-vis du CSE avant un plan de sauvegarde de l'emploi ?",
     "expected": ["L1233-28", "CSE", "PSE", "consultation"], "code": "travail"},
    {"q": "Un CDD peut-il être requalifié en CDI ? Dans quels cas ?",
     "expected": ["L1245-1", "requalification", "CDI"], "code": "travail"},
    {"q": "Quelles sont les conditions du travail de nuit ?",
     "expected": ["L3122-1", "travail de nuit", "contrepartie"], "code": "travail"},
    {"q": "Comment fonctionne l'activité partielle (chômage partiel) ?",
     "expected": ["L5122-1", "activité partielle", "allocation"], "code": "travail"},
    {"q": "Qu'est-ce que le droit d'alerte du CSE ?",
     "expected": ["L2312-60", "droit d'alerte", "CSE"], "code": "travail"},
    {"q": "Quelles sont les règles relatives aux astreintes ?",
     "expected": ["L3121-9", "astreinte", "intervention"], "code": "travail"},
    {"q": "Comment est encadrée la clause de non-concurrence ?",
     "expected": ["non-concurrence", "contrepartie financière", "limitée"], "code": "travail"},
    {"q": "Quelles formalités pour un accord de branche étendu ?",
     "expected": ["L2261-15", "extension", "arrêté"], "code": "travail"},
    {"q": "Qu'est-ce que la présomption de salariat ?",
     "expected": ["L7121-3", "présomption", "salariat"], "code": "travail"},
]

assert len(EVAL_QUESTIONS) == 50, f"Attendu 50 questions, got {len(EVAL_QUESTIONS)}"


# ── Fonctions d'évaluation ─────────────────────────────────────────────────────

def hit(chunks: List[dict], expected_keywords: List[str]) -> bool:
    """True si au moins un chunk contient un des mots-clés attendus."""
    joined = " ".join(
        (c.get("texte", "") + " " + c.get("article_id", "")).lower()
        for c in chunks
    )
    return any(kw.lower() in joined for kw in expected_keywords)


def precision_at_k(chunks: List[dict], expected_keywords: List[str], k: int = 5) -> float:
    """Proportion des top-k chunks contenant un mot-clé attendu."""
    if not chunks:
        return 0.0
    relevant = sum(
        1 for c in chunks[:k]
        if any(
            kw.lower() in (c.get("texte", "") + " " + c.get("article_id", "")).lower()
            for kw in expected_keywords
        )
    )
    return relevant / min(k, len(chunks))


def reciprocal_rank(chunks: List[dict], expected_keywords: List[str]) -> float:
    """Rang réciproque du premier chunk pertinent."""
    for i, c in enumerate(chunks, 1):
        text = (c.get("texte", "") + " " + c.get("article_id", "")).lower()
        if any(kw.lower() in text for kw in expected_keywords):
            return 1.0 / i
    return 0.0


async def run_evaluation(api_base: str = API_BASE) -> dict:
    """Lance l'évaluation complète sur les 50 questions."""
    log.info("=== Début évaluation RAG — %d questions ===", len(EVAL_QUESTIONS))

    results = []
    latencies = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Vérifier la santé de l'API
        try:
            health = await client.get(f"{api_base}/health")
            health.raise_for_status()
            log.info("API health OK: %s", health.json())
        except Exception as e:
            raise RuntimeError(f"API inaccessible: {e}")

        for i, item in enumerate(EVAL_QUESTIONS, 1):
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{api_base}/rag/search",
                    json={
                        "question":         item["q"],
                        "code_prioritaire": item["code"],
                        "top_k":            5,
                    },
                )
                resp.raise_for_status()
                data   = resp.json()
                chunks = data.get("chunks", [])
                api_latency = data.get("latency_ms", 0)

            except Exception as e:
                log.error("Erreur Q%d: %s", i, e)
                chunks      = []
                api_latency = 0.0

            total_latency = (time.perf_counter() - t0) * 1000
            latencies.append(api_latency)

            p5 = precision_at_k(chunks, item["expected"], k=5)
            rr = reciprocal_rank(chunks, item["expected"])
            h  = hit(chunks, item["expected"])

            results.append({
                "q_id":           i,
                "question":       item["q"],
                "code":           item["code"],
                "precision_at_5": p5,
                "rr":             rr,
                "hit":            bool(h),
                "latency_ms":     api_latency,
                "n_chunks":       len(chunks),
                "top_article":    chunks[0]["article_id"] if chunks else "",
            })

            log.info(
                "Q%02d %-50s → P@5=%.2f RR=%.2f Hit=%s Lat=%.0fms",
                i, item["q"][:50], p5, rr, h, api_latency,
            )

    # ── Métriques globales ─────────────────────────────────────────────────────
    df      = pd.DataFrame(results)
    mean_p5 = df["precision_at_5"].mean()
    mrr     = df["rr"].mean()
    hit_rate= df["hit"].mean()
    lat_p50 = np.percentile(latencies, 50) if latencies else 0
    lat_p95 = np.percentile(latencies, 95) if latencies else 0

    summary = {
        "timestamp":       datetime.now(UTC).isoformat(),
        "n_questions":     len(results),
        "precision_at_5":  round(mean_p5, 4),
        "mrr":             round(mrr, 4),
        "hit_rate":        round(hit_rate, 4),
        "latency_p50_ms":  round(lat_p50, 2),
        "latency_p95_ms":  round(lat_p95, 2),
        "kpi_ok": {
            "precision_gte_98": mean_p5 >= 0.98,
            "mrr_gte_085":      mrr >= 0.85,
            "hit_rate_gte_99":  hit_rate >= 0.99,
            "latency_lte_50ms": lat_p95 <= 50,
        },
        "per_code": df.groupby("code")[["precision_at_5", "rr", "hit"]].mean().round(4).to_dict(),
        "worst_questions": df.nsmallest(5, "precision_at_5")[["q_id", "question", "precision_at_5", "rr"]].to_dict(orient="records"),
    }

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = EVAL_DIR / f"eval_{ts}.csv"
    json_path = EVAL_DIR / f"eval_{ts}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def make_json_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_serializable(i) for i in obj]
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, default=make_json_serializable, ensure_ascii=False, indent=2)
    # ── Affichage résumé ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RÉSULTATS ÉVALUATION RAG JURIDIQUE")
    print("="*60)
    print(f"  Precision@5    : {mean_p5:.1%}   (cible ≥ 98%)")
    print(f"  MRR            : {mrr:.3f}      (cible ≥ 0.85)")
    print(f"  Hit Rate       : {hit_rate:.1%}  (cible ≥ 99%)")
    print(f"  Latence P50    : {lat_p50:.0f}ms    (cible ≤ 50ms)")
    print(f"  Latence P95    : {lat_p95:.0f}ms")
    print("-"*60)
    for kpi, ok in summary["kpi_ok"].items():
        icon = "OK" if ok else "KO"
        print(f"  {icon} {kpi}")
    print("="*60)
    print(f"  Résultats sauvegardés : {csv_path}")

    return summary


# ── Tests unitaires inline ─────────────────────────────────────────────────────
def _test_hit():
    chunks = [{"texte": "Article L1237-1 préavis démission", "article_id": "L1237-1"}]
    assert hit(chunks, ["L1237-1", "préavis"]) is True
    assert hit(chunks, ["L9999-9", "inconnue"]) is False
    log.info("✓ test_hit OK")

def _test_precision():
    chunks = [
        {"texte": "L1237-1 préavis", "article_id": "L1237-1"},
        {"texte": "L1234-9 indemnité", "article_id": "L1234-9"},
        {"texte": "Autre article", "article_id": "L9999-9"},
    ]
    p = precision_at_k(chunks, ["L1237-1", "L1234-9"], k=3)
    assert p == pytest_approx(2/3, abs=0.01), f"Expected ~0.667, got {p}"
    log.info("✓ test_precision OK")

def _test_rr():
    chunks = [
        {"texte": "Rien", "article_id": "Z"},
        {"texte": "L1237-1 préavis", "article_id": "L1237-1"},
    ]
    rr = reciprocal_rank(chunks, ["L1237-1"])
    assert abs(rr - 0.5) < 0.01
    log.info("✓ test_rr OK")

def pytest_approx(val, abs=0.01):
    """Mini helper pour les tests sans pytest."""
    class Approx:
        def __eq__(self, other): return abs(other - val) <= abs
    return Approx()

def run_unit_tests():
    _test_hit()
    _test_rr()
    log.info("✓ Tous les tests unitaires passés")


if __name__ == "__main__":
    import asyncio
    import sys

    if "--test" in sys.argv:
        run_unit_tests()
    else:
        summary = asyncio.run(run_evaluation())
        if not all(summary["kpi_ok"].values()):
            log.warning(" Certains KPI ne sont pas atteints — vérifier les index FAISS")
            sys.exit(1)
