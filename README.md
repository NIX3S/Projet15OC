#  RAG Juridique 

Système RAG offline pour le droit français : 4 codes + jurisprudence Cour de cassation.

```
Légifrance API/XML → Airflow → FAISS IndexFlatL2(768) → FastAPI → Streamlit
                                    ↓
                             SQLite (par code)
```

##  Structure

```
legal_rag_project/
├── dags/legal_codes_daily.py          # Airflow DAG deltas quotidiens + FAISS
├── api/rag_legal_api.py               # FastAPI RAG service (search + chat + health)
├── eval/evaluation_rag.py             # 50 questions Precision@5 + MRR
├── dashboard/
│   ├── streamlit_monitoring.py        #  Dashboard métriques (port 8501)
│   └── streamlit_chat_interface.py    #  Chat ChatGPT-style (port 8502)
├── data/bootstrap.py                  # XML/API → SQLite → FAISS initial
├── docker-compose.yml
├── requirements.txt
└── README.md
```

##  Démarrage rapide

### 1. Prérequis

```bash
# Docker + Docker Compose
docker --version        # ≥ 24
docker compose version  # ≥ 2.20

# Ou local (Python 3.11+)
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Éditer .env avec vos credentials PISTE Légifrance
```

**.env** :
```env
LEGIFRANCE_CLIENT_ID=your_client_id
LEGIFRANCE_CLIENT_SECRET=your_client_secret
SLACK_WEBHOOK_URL=https://hooks.slack.com/...   # optionnel
AIRFLOW_FERNET_KEY=zPmEyyEFUQ7uF5pNBIPqfOFaYv7SyKsH2JCfRmI9d2k=
```

> Credentials PISTE : https://developer.aife.economie.gouv.fr/

### 3. Bootstrap initial (1 fois)

```bash
# Via Docker
docker compose --profile bootstrap up bootstrap

# Ou local
export RAG_DATA_DIR=./data
python data/bootstrap.py --codes travail securite civil proc_civile --test
```

### 4. Démarrer les services

```bash
docker compose up -d
```

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8501 | Métriques, alertes, recherche |
| Chat | http://localhost:8502 | Interface conversationnelle |
| API FastAPI | http://localhost:8000/docs | Swagger RAG |
| CRON |  | DAG scheduling |
| Ollama | http://localhost:11434 | Mistral Nemo LLM |

---

##  Sources Légifrance

| Code | LEGITEXT | URL |
|------|----------|-----|
| Travail | `LEGITEXT000006072050` | https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006072050/ |
| Sécurité sociale | `LEGITEXT000006073189` | https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006073189/ |
| Civil | `LEGITEXT000006070721` | https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006070721/ |
| Proc. Civile | `LEGITEXT000006070716` | https://www.legifrance.gouv.fr/codes/texte_lc/LEGITEXT000006070716/ |
| Jurisprudence | Cour cassation chambre sociale | Travail uniquement |

---

##  Architecture technique

### Pipeline RAG

```
Question → MiniLM-L12-v2 (embed) → FAISS IndexFlatL2(768) → top-5 chunks
                                                                    ↓
                                              Prompt IRAC + historique (8k tokens)
                                                                    ↓
                                                          Mistral Nemo → Réponse
```

### Modèle d'embeddings

- **`paraphrase-multilingual-MiniLM-L12-v2`** — optimisé français, dim=384
- Normalisation L2 avant indexation FAISS
- Distance L2 (plus petit = plus pertinent)

### Context window

- Historique : 10 derniers messages conservés
- Fenêtre LLM : 8 192 tokens (Mistral Nemo)
- Format IRAC : ISSUE / RULE / ANALYSIS / CONCLUSION / REFERENCES

---

##  KPI production

| Métrique | Cible | Mesure |
|----------|-------|--------|
| Precision@5 | ≥ 98% | `eval/evaluation_rag.py` |
| MRR | ≥ 0.85 | `eval/evaluation_rag.py` |
| Hit Rate | ≥ 99% | `eval/evaluation_rag.py` |
| Latence FAISS P95 | ≤ 50ms | `/health` endpoint |
| Détection abrogations | 100% | Filtre statut + alerte |
| Context window | 8k tokens | Truncature historique |

---

##  Commandes utiles

```bash
# Lancer l'évaluation RAG
python eval/evaluation_rag.py

# Tests unitaires
python eval/evaluation_rag.py --test

# Forcer la mise à jour d'un code
docker compose exec airflow airflow dags trigger legal_codes_daily

# Recharger un index FAISS dans l'API
curl -X POST http://localhost:8000/admin/reload/travail

# Voir les logs API
docker compose logs -f api

# Status FAISS
curl http://localhost:8000/health | python -m json.tool
```

---

##  Pièges connus et solutions

| Problème | Cause | Solution |
|----------|-------|---------|
| Articles fantômes ABROGÉS | Delta partiel | `df_base[df_base.id != delta.id]` dans DAG |
| Context window overflow | Historique illimité | Truncature à 10 messages |
| FAISS corrompu | Crash mid-write | `rebuild_index()` au startup API |
| Precision < 95% | Index périmé | Alerte Streamlit rouge + retrigger DAG |
| Jurisprudence tous codes | Bug scope | `code="jurisprudence"` filtré dans API |
| Session state perdu | Rechargement Streamlit | `st.session_state.setdefault()` |

---

##  Schéma base SQLite

```sql
CREATE TABLE articles (
    id           TEXT PRIMARY KEY,
    texte        TEXT NOT NULL,
    statut       TEXT NOT NULL,  -- VIGUEUR | ABROGE | MODIFIE
    date_vigueur TEXT,
    nor          TEXT,
    metadata     TEXT,           -- JSON : code, id_nor, idcc, juridiction...
    code         TEXT NOT NULL
);
CREATE INDEX idx_statut ON articles(statut);
CREATE INDEX idx_id     ON articles(id);
```

---

##  Metadata FAISS (par vecteur)

```python
{
    "code":        "travail|securite|civil|proc_civile|jurisprudence",
    "id_nor":      "NOR:MTLSXXXXXXXXXX",
    "statut":      "VIGUEUR|ABROGE|MODIFIE",
    "date_vigueur": "2026-03-12",
    "idcc":        "2216|null",
    "juridiction": "Cour_cassation|null",
    "article_id":  "L1234-1|R567-2"
}
```

---

##  Auteur

**Paul Lesage** — Data Scientist ML/AI  
2026
