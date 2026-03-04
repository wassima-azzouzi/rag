# 🤖 VaultAI - Chatbot RAG Premium

Assistant intelligent basé sur le pipeline **RAG (Retrieval Augmented Generation)** utilisant **ChromaDB** et **Groq (LLaMA 3.1)**.

## 🚀 Fonctionnalités
- **Persistence** : Stockage vectoriel persistant via ChromaDB (dossier `./chroma_db`).
- **Interface Premium** : Design moderne avec bulles de chat animées et glassmorphism.
- **Réglages IA** : Température et Top-K ajustables en temps réel.
- **Pipeline Transparent** : Visualisation des étapes d'indexation et citation des sources.
- **Paramètres de Chunking** : Configuration personnalisée du découpage des documents.

## 🛠️ Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/wassima-azzouzi/rag.git
   cd rag
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration** :
   Créez un fichier `.env` à la racine :
   ```env
   GROQ_API_KEY=votre_cle_groq
   ```

## 🏃 Lancement
```bash
streamlit run app.py
```

## 📖 Utilisation
1. Chargez vos fichiers PDF dans la barre latérale.
2. Réglez la taille des morceaux (Chunks) si besoin.
3. Cliquez sur **✨ Indexer**.
4. Posez vos questions !
