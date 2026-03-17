"""
DREETS-IA — Interface Chat Conversationnel (ChatGPT/Perplexity style)
Context window 8k tokens | RAG FAISS | Mistral Nemo | Export JSON
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")

ALL_CODES = ["travail", "securite", "civil", "proc_civile", "jurisprudence"]
CODE_LABELS = {
    "travail":       "⚒️ Code du Travail",
    "securite":      "🏥 Code Sécu Sociale",
    "civil":         "📜 Code Civil",
    "proc_civile":   "⚖️ Code Proc. Civile",
    "jurisprudence": "🏛️ Jurisprudence Cass.",
}

st.set_page_config(
    page_title="DREETS-IA Chat Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ChatGPT-style ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Messages */
.user-bubble {
    background: #2563eb22;
    border-left: 3px solid #2563eb;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
.bot-bubble {
    background: #16213022;
    border-left: 3px solid #9b59b6;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
.source-chip {
    display: inline-block;
    background: #1e1e3022;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75em;
    margin: 2px;
    cursor: pointer;
}
.abroge-warning {
    background: #c0392b22;
    border: 1px solid #e74c3c;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 0.8em;
    color: #e74c3c;
    margin: 4px 0;
}
.latency-badge {
    font-size: 0.72em;
    color: #888;
    float: right;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("code_prioritaire", "travail")
st.session_state.setdefault("top_k", 5)
st.session_state.setdefault("show_sources", True)
st.session_state.setdefault("total_queries", 0)
st.session_state.setdefault("total_latency_ms", 0.0)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ DREETS-IA")
    st.caption("Chat Juridique · Hauts-de-France 2026")
    st.divider()

    st.markdown("### ⚙️ Paramètres RAG")
    code_sel = st.selectbox(
        "Code prioritaire",
        options=ALL_CODES,
        format_func=lambda x: CODE_LABELS[x],
        index=ALL_CODES.index(st.session_state.code_prioritaire),
    )
    st.session_state.code_prioritaire = code_sel

    top_k = st.slider(
        "Nombre d'articles (top_k)", min_value=1, max_value=10,
        value=st.session_state.top_k,
    )
    st.session_state.top_k = top_k

    show_sources = st.toggle("Afficher les sources RAG", value=st.session_state.show_sources)
    st.session_state.show_sources = show_sources

    st.divider()

    # Stats session
    n_q  = st.session_state.total_queries
    lat  = st.session_state.total_latency_ms
    avg  = lat / n_q if n_q else 0
    st.caption(f"📊 **Session**")
    st.caption(f"Questions : {n_q}")
    st.caption(f"Latence moy. : {avg:.0f}ms")

    st.divider()

    # Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries    = 0
            st.session_state.total_latency_ms = 0.0
            st.rerun()
    with col_b:
        # Export JSON
        if st.session_state.messages:
            export_data = json.dumps(
                {
                    "export_date": datetime.utcnow().isoformat(),
                    "codes": [st.session_state.code_prioritaire],
                    "messages": st.session_state.messages,
                },
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                "💾 Export",
                data=export_data,
                file_name=f"dreets_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.caption("**Codes disponibles**")
    for code, label in CODE_LABELS.items():
        st.caption(label)

    st.divider()
    st.caption("⚡ Powered by Mistral Nemo + FAISS")
    st.caption("📚 Sources : Légifrance 2026")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("⚖️ DREETS-IA — Assistant Juridique")
st.caption(f"Code prioritaire actif : **{CODE_LABELS[st.session_state.code_prioritaire]}** | Top-{st.session_state.top_k} sources")

# Message de bienvenue
if not st.session_state.messages:
    st.markdown("""
    <div class='bot-bubble'>
    🤖 <b>DREETS-IA</b> — Bonjour Paul ! Je suis votre assistant juridique spécialisé en droit français (2026).<br><br>
    Je peux vous aider sur :
    <ul>
      <li>⚒️ <b>Code du Travail</b> — licenciement, contrats, temps de travail, conventions collectives</li>
      <li>🏥 <b>Sécurité sociale</b> — cotisations, indemnités, retraite</li>
      <li>📜 <b>Code Civil</b> — contrats, responsabilité, prescription</li>
      <li>⚖️ <b>Procédure Civile</b> — délais, compétences, saisine</li>
      <li>🏛️ <b>Jurisprudence</b> — arrêts Cour de cassation (chambre sociale)</li>
    </ul>
    <i>Posez votre question juridique ci-dessous.</i>
    </div>
    """, unsafe_allow_html=True)

# Affichage de l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>👤 <b>Paul</b><br>{msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        # Réponse bot
        answer  = msg.get("content", "")
        sources = msg.get("sources", [])
        latency = msg.get("latency_ms", 0)

        st.markdown(
            f"<div class='bot-bubble'>"
            f"🤖 <b>DREETS-IA</b>"
            f"<span class='latency-badge'>⏱️ {latency:.0f}ms</span><br><br>"
            f"{answer}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Sources RAG
        if show_sources and sources:
            with st.expander(f"📚 Sources RAG ({len(sources)} articles)", expanded=False):
                for s in sources:
                    statut_icon = "✅" if s["statut"] == "VIGUEUR" else "⚠️ ABROGÉ"
                    if s["statut"] == "ABROGE":
                        st.markdown(
                            f"<div class='abroge-warning'>⚠️ {s['article_id']} ({s['code']}) — ABROGÉ — Ne pas utiliser</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        with st.container():
                            c1, c2, c3 = st.columns([2, 1, 1])
                            c1.caption(f"**{s['article_id']}** ({CODE_LABELS.get(s['code'], s['code'])})")
                            c2.caption(f"NOR: {s['nor'] or 'N/A'}")
                            c3.caption(f"{statut_icon}")
                            if s.get("idcc"):
                                st.caption(f"IDCC: {s['idcc']}")
                            st.caption(s["texte"][:300] + "..." if len(s.get("texte","")) > 300 else s.get("texte",""))
                            st.divider()


# ── Chat input ─────────────────────────────────────────────────────────────────
if question := st.chat_input("Posez votre question juridique... (ex: préavis démission CDI 3 ans ancienneté)"):

    # Ajouter la question à l'historique (session state)
    st.session_state.messages.append({"role": "user", "content": question})

    # Afficher immédiatement la question
    st.markdown(
        f"<div class='user-bubble'>👤 <b>Paul</b><br>{question}</div>",
        unsafe_allow_html=True,
    )

    # Préparer l'historique pour l'API (10 derniers messages)
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
        if m["role"] in ("user", "assistant")
    ]

    # Appel API RAG
    with st.spinner("🔍 Recherche juridique en cours..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/rag/chat",
                json={
                    "question":        question,
                    "codes":           [st.session_state.code_prioritaire],  # <-- ici
                    "top_k":           st.session_state.top_k,
                    "context_history": api_history,
                },
                timeout=90.0,
            )
            resp.raise_for_status()
            data = resp.json()

            answer  = data.get("answer", "Désolé, je n'ai pas pu générer de réponse.")
            sources = data.get("chunks", [])
            latency = data.get("latency_ms", 0)

        except httpx.TimeoutException:
            answer  = "⏱️ Délai dépassé (>90s). L'API est peut-être surchargée. Réessayez dans quelques instants."
            sources = []
            latency = 0
        except httpx.HTTPStatusError as e:
            answer  = f"❌ Erreur API ({e.response.status_code}): {e.response.text[:200]}"
            sources = []
            latency = 0
        except Exception as e:
            answer  = f"❌ Erreur inattendue: {str(e)}"
            sources = []
            latency = 0

    # Ajouter la réponse à l'historique
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "sources":    sources,
        "latency_ms": latency,
    })

    # Stats session
    st.session_state.total_queries    += 1
    st.session_state.total_latency_ms += latency

    # Afficher la réponse
    st.markdown(
        f"<div class='bot-bubble'>"
        f"🤖 <b>DREETS-IA</b>"
        f"<span class='latency-badge'>⏱️ {latency:.0f}ms</span><br><br>"
        f"{answer}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Sources
    if show_sources and sources:
        with st.expander(f"📚 Sources RAG ({len(sources)} articles)", expanded=True):
            abroge_list = [s for s in sources if s["statut"] == "ABROGE"]
            vigueur_list = [s for s in sources if s["statut"] != "ABROGE"]

            if abroge_list:
                st.warning(f"⚠️ {len(abroge_list)} article(s) ABROGÉ(s) dans les résultats — référez-vous aux textes VIGUEUR uniquement")

            for s in vigueur_list:
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.caption(f"**{s['article_id']}** ({CODE_LABELS.get(s['code'], s['code'])})")
                c2.caption(f"NOR: {s['nor'] or 'N/A'}")
                c3.caption(f"Score: {s['score']:.4f}")
                if s.get("idcc"):
                    st.caption(f"IDCC: {s['idcc']}")
                st.caption(s["texte"][:400] + "..." if len(s.get("texte","")) > 400 else s.get("texte",""))
                st.divider()

    st.rerun()
