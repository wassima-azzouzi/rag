# RAG Chat PDF – Assistant style ChatGPT

Ce projet est une application Streamlit qui permet d'interroger des fichiers PDF
(cours, articles, rapports, etc.) à l'aide d'un pipeline RAG (Retrieval Augmented Generation)
basé sur :
- SentenceTransformer (CamemBERT) pour les embeddings
- FAISS pour la recherche vectorielle
- Groq (LLaMA 3) pour la génération de réponse

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows

pip install -r requirements.txt
```

## Configuration

Crée un fichier `.env` ou exporte la variable d'environnement suivante :

```bash
export GROQ_API_KEY=TA_CLE_GROQ_ICI
```

ou sous Windows (PowerShell) :

```powershell
$env:GROQ_API_KEY="TA_CLE_GROQ_ICI"
```

## Lancer l'application

```bash
streamlit run app_rag_chat.py
```

Ensuite, ouvre l'URL locale fournie par Streamlit dans ton navigateur.

1. Charge un ou plusieurs PDF dans la barre latérale
2. Clique sur **Construire / Recharger l'index**
3. Pose ta question dans la zone de texte (style ChatGPT)
