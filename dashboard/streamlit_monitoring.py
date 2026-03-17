"""
DREETS-IA — Dashboard de monitoring RAG Juridique
Métriques : VIGUEUR/ABROGE, Precision@5, Latence FAISS, Alertes
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")
DATA_DIR  = Path(os.getenv("RAG_DATA_DIR", "/data"))
EVAL_DIR  = Path("eval/results")

ALL_CODES = ["travail", "securite", "civil", "proc_civile", "jurisprudence"]
CODE_LABELS = {
    "travail":     "Code du Travail",
    "securite":    "Code Sécurité Sociale",
    "civil":       "Code Civil",
    "proc_civile": "Code Proc. Civile",
    "jurisprudence": "Jurisprudence Cass.",
}
COLORS = {
    "VIGUEUR": "#2ecc71",
    "ABROGE":  "#e74c3c",
    "MODIFIE": "#f39c12",
}

st.set_page_config(
    page_title="DREETS-IA Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background:#1a1a2e; border-radius:8px; padding:16px; margin:4px 0; }
.alert-red   { background:#c0392b22; border-left:4px solid #e74c3c; padding:12px; border-radius:4px; }
.alert-green { background:#27ae6022; border-left:4px solid #2ecc71; padding:12px; border-radius:4px; }
.kpi-ok   { color:#2ecc71; font-weight:bold; }
.kpi-fail { color:#e74c3c; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_health() -> dict:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e), "codes": {}}


@st.cache_data(ttl=300)
def load_sqlite_stats(code_key: str) -> dict:
    db_path = DATA_DIR / f"{code_key}.db"
    if not db_path.exists():
        return {"VIGUEUR": 0, "ABROGE": 0, "MODIFIE": 0, "total": 0}
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(
                "SELECT statut, COUNT(*) as n FROM articles GROUP BY statut", conn
            )
        counts = df.set_index("statut")["n"].to_dict()
        counts["total"] = sum(counts.values())
        return counts
    except Exception:
        return {"VIGUEUR": 0, "ABROGE": 0, "MODIFIE": 0, "total": 0}


@st.cache_data(ttl=300)
def load_latest_eval() -> dict:
    if not EVAL_DIR.exists():
        return {}
    jsons = sorted(EVAL_DIR.glob("eval_*.json"), reverse=True)
    if not jsons:
        return {}
    with open(jsons[0], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_daily_stats() -> dict:
    stats_path = DATA_DIR / "daily_stats.json"
    if not stats_path.exists():
        return {}
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def semantic_search(question: str, code: str, top_k: int = 5) -> dict:
    try:
        resp = httpx.post(
            f"{API_BASE}/rag/search",
            json={"question": question, "code_prioritaire": code, "top_k": top_k},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "chunks": [], "latency_ms": 0}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/3/3f/Logo_Dreets.png",
             width=160, use_column_width=False)
    st.markdown("## ⚖️ DREETS-IA")
    st.caption("Monitoring RAG Juridique — 2026")
    st.divider()

    if st.button("🔄 Rafraîchir tout", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        time.sleep(1)
        st.rerun()

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Métriques", "🔍 Recherche sémantique", "⚠️ Alertes & Audit"],
                    label_visibility="collapsed")


# ── Données ────────────────────────────────────────────────────────────────────
health      = fetch_health()
latest_eval = load_latest_eval()
daily_stats = load_daily_stats()

all_stats = {c: load_sqlite_stats(c) for c in ALL_CODES}

# ── Page : Métriques ───────────────────────────────────────────────────────────
if page == "📊 Métriques":
    st.title("📊 Dashboard RAG Juridique — DREETS Hauts-de-France")
    st.caption(f"Dernière mise à jour : {health.get('timestamp', 'N/A')}")

    # ── KPI Banner ────────────────────────────────────────────────────────────
    kpi_cols = st.columns(4)

    total_vigueur = sum(s.get("VIGUEUR", 0) for s in all_stats.values())
    total_abroge  = sum(s.get("ABROGE", 0) for s in all_stats.values())
    p5_val        = latest_eval.get("precision_at_5", 0)
    mrr_val       = latest_eval.get("mrr", 0)

    with kpi_cols[0]:
        st.metric("Articles VIGUEUR", f"{total_vigueur:,}", delta=None)
    with kpi_cols[1]:
        st.metric("Articles ABROGÉS", f"{total_abroge:,}",
                  delta=None, delta_color="inverse")
    with kpi_cols[2]:
        color_p5 = "normal" if p5_val >= 0.98 else "inverse"
        st.metric("Precision@5", f"{p5_val:.1%}" if p5_val else "N/A",
                  delta="≥98% cible", delta_color=color_p5)
    with kpi_cols[3]:
        color_mrr = "normal" if mrr_val >= 0.85 else "inverse"
        st.metric("MRR", f"{mrr_val:.3f}" if mrr_val else "N/A",
                  delta="≥0.85 cible", delta_color=color_mrr)

    st.divider()

    # ── Statuts par code ──────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📚 Articles par code")
        rows = []
        for code_key, label in CODE_LABELS.items():
            s = all_stats[code_key]
            rows.append({
                "Code": label,
                "VIGUEUR": s.get("VIGUEUR", 0),
                "ABROGE":  s.get("ABROGE", 0),
                "MODIFIE": s.get("MODIFIE", 0),
                "Total":   s.get("total", 0),
            })
        df_table = pd.DataFrame(rows)
        st.dataframe(
            df_table.style.format({"VIGUEUR": "{:,}", "ABROGE": "{:,}", "Total": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.subheader("🥧 Répartition statuts")
        pie_data = []
        for code_key, label in CODE_LABELS.items():
            s = all_stats[code_key]
            for statut in ["VIGUEUR", "ABROGE", "MODIFIE"]:
                n = s.get(statut, 0)
                if n > 0:
                    pie_data.append({"Code": label, "Statut": statut, "n": n})
        if pie_data:
            df_pie = pd.DataFrame(pie_data)
            fig = px.bar(df_pie, x="Code", y="n", color="Statut",
                         color_discrete_map=COLORS, barmode="stack",
                         height=300, title="Articles par code et statut")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── KPI Évaluation ────────────────────────────────────────────────────────
    st.subheader("🎯 KPI Évaluation RAG")
    if latest_eval:
        kpis = latest_eval.get("kpi_ok", {})
        eval_cols = st.columns(4)
        kpi_items = [
            ("Precision@5 ≥ 98%", "precision_gte_98"),
            ("MRR ≥ 0.85",        "mrr_gte_085"),
            ("Hit Rate ≥ 99%",    "hit_rate_gte_99"),
            ("Latence ≤ 50ms",    "latency_lte_50ms"),
        ]
        for col, (label, key) in zip(eval_cols, kpi_items):
            ok = kpis.get(key, False)
            col.markdown(
                f"<div class='metric-card'><span class='{'kpi-ok' if ok else 'kpi-fail'}'>"
                f"{'✅' if ok else '❌'} {label}</span></div>",
                unsafe_allow_html=True,
            )

        # Par code
        per_code = latest_eval.get("per_code", {})
        if per_code and "precision_at_5" in per_code:
            p5_per_code = pd.DataFrame({
                "Code":        list(per_code["precision_at_5"].keys()),
                "Precision@5": list(per_code["precision_at_5"].values()),
                "MRR":         [per_code.get("rr", {}).get(c, 0)
                                 for c in per_code["precision_at_5"].keys()],
            })
            fig2 = go.Figure()
            fig2.add_bar(x=p5_per_code["Code"], y=p5_per_code["Precision@5"],
                         name="Precision@5", marker_color="#3498db")
            fig2.add_bar(x=p5_per_code["Code"], y=p5_per_code["MRR"],
                         name="MRR", marker_color="#9b59b6")
            fig2.add_hline(y=0.98, line_dash="dash", line_color="#2ecc71",
                            annotation_text="Cible P@5 98%")
            fig2.add_hline(y=0.85, line_dash="dot", line_color="#f39c12",
                            annotation_text="Cible MRR 0.85")
            fig2.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0),
                                paper_bgcolor="rgba(0,0,0,0)", legend_orientation="h")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Aucun résultat d'évaluation disponible. Lancer `python eval/evaluation_rag.py`")

    # ── FAISS status ──────────────────────────────────────────────────────────
    st.subheader("🗄️ Index FAISS")
    faiss_cols = st.columns(len(ALL_CODES))
    for col, code_key in zip(faiss_cols, ALL_CODES):
        code_health = health.get("codes", {}).get(code_key, {})
        vectors     = code_health.get("faiss_vectors", 0)
        loaded      = code_health.get("index_loaded", False)
        icon = "✅" if loaded and vectors > 0 else "❌"
        col.metric(CODE_LABELS[code_key], f"{vectors:,} vec.", delta=f"{icon} {'OK' if loaded else 'KO'}")


# ── Page : Recherche sémantique ───────────────────────────────────────────────
elif page == "🔍 Recherche sémantique":
    st.title("🔍 Recherche sémantique dans les codes")

    with st.form("search_form"):
        q_col, opts_col = st.columns([3, 1])
        with q_col:
            question = st.text_input("Question juridique", placeholder="Ex: durée préavis démission CDI")
        with opts_col:
            code_sel = st.selectbox("Code prioritaire", options=ALL_CODES,
                                     format_func=lambda x: CODE_LABELS[x])
            top_k    = st.slider("Top K", 1, 10, 5)
        submitted = st.form_submit_button("🔎 Rechercher", use_container_width=True)

    if submitted and question:
        with st.spinner("Recherche en cours..."):
            result = semantic_search(question, code_sel, top_k)

        if "error" in result:
            st.error(f"Erreur API: {result['error']}")
        else:
            chunks    = result.get("chunks", [])
            latency   = result.get("latency_ms", 0)
            st.caption(f"⏱️ Latence FAISS : {latency:.0f}ms | {len(chunks)} résultats")

            if not chunks:
                st.warning("Aucun article trouvé pour cette question.")
            for i, c in enumerate(chunks, 1):
                statut_color = "🟢" if c["statut"] == "VIGUEUR" else "🔴"
                with st.expander(
                    f"{i}. {c['article_id']} ({CODE_LABELS.get(c['code'], c['code'])}) "
                    f"{statut_color} {c['statut']} — Score: {c['score']:.4f}"
                ):
                    if c["statut"] == "ABROGE":
                        st.error("⚠️ Article ABROGÉ — Ne pas utiliser comme référence juridique")
                    cols = st.columns(3)
                    cols[0].caption(f"**NOR:** {c['nor'] or 'N/A'}")
                    cols[1].caption(f"**IDCC:** {c.get('idcc') or 'N/A'}")
                    cols[2].caption(f"**En vigueur:** {c['date_vigueur'] or 'N/A'}")
                    st.markdown(c["texte"])


# ── Page : Alertes & Audit ────────────────────────────────────────────────────
elif page == "⚠️ Alertes & Audit":
    st.title("⚠️ Alertes & Audit Qualité")

    # Alertes critiques
    alerts = []

    p5_val  = latest_eval.get("precision_at_5", 1.0)
    mrr_val = latest_eval.get("mrr", 1.0)

    if p5_val and p5_val < 0.95:
        alerts.append(("CRITIQUE", f"Precision@5 = {p5_val:.1%} — En dessous du seuil critique 95%"))
    elif p5_val and p5_val < 0.98:
        alerts.append(("WARNING", f"Precision@5 = {p5_val:.1%} — En dessous de la cible 98%"))

    for code_key in ALL_CODES:
        s = all_stats[code_key]
        abroge  = s.get("ABROGE", 0)
        vigueur = s.get("VIGUEUR", 0)
        if abroge > 0 and vigueur > 0 and (abroge / vigueur) > 0.1:
            alerts.append(("WARNING", f"{CODE_LABELS[code_key]} : {abroge} articles ABROGÉS ({abroge/vigueur:.0%} du total)"))

    if health.get("status") == "error":
        alerts.append(("CRITIQUE", f"API inaccessible : {health.get('error', 'Erreur inconnue')}"))

    if not alerts:
        st.markdown(
            "<div class='alert-green'>✅ Aucune alerte critique — Système nominal</div>",
            unsafe_allow_html=True
        )
    else:
        for level, msg in alerts:
            icon = "🔴" if level == "CRITIQUE" else "🟡"
            st.markdown(
                f"<div class='alert-red'>{icon} <b>{level}</b> — {msg}</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # Pires questions éval
    st.subheader("📉 Questions difficiles (dernière évaluation)")
    worst = latest_eval.get("worst_questions", [])
    if worst:
        df_worst = pd.DataFrame(worst)
        st.dataframe(df_worst, use_container_width=True, hide_index=True)
    else:
        st.info("Lancer `python eval/evaluation_rag.py` pour peupler cet audit.")

    st.divider()

    # Historique évals
    st.subheader("📈 Historique Precision@5 + MRR")
    if EVAL_DIR.exists():
        evals = []
        for jf in sorted(EVAL_DIR.glob("eval_*.json")):
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            evals.append({
                "Date": d.get("timestamp", "")[:10],
                "P@5":  d.get("precision_at_5", 0),
                "MRR":  d.get("mrr", 0),
            })
        if evals:
            df_hist = pd.DataFrame(evals)
            fig = px.line(df_hist, x="Date", y=["P@5", "MRR"], markers=True,
                          title="Évolution des métriques RAG", height=300)
            fig.add_hline(y=0.98, line_dash="dash", line_color="#2ecc71",
                           annotation_text="Cible P@5")
            fig.add_hline(y=0.85, line_dash="dot",  line_color="#f39c12",
                           annotation_text="Cible MRR")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun historique d'évaluations disponible.")
