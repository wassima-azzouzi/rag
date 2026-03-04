import os
import tempfile
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement (.env)
load_dotenv()

# -------------------------------------------------------------
# CONFIGURATION & STYLE CSS PREMIUM
# -------------------------------------------------------------
st.set_page_config(page_title="VaultAI - Assistant RAG Premium", layout="wide", page_icon="🤖")

def local_css():
    st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e);
        color: #e9ecef;
    }
    
    /* Input Area */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Chat Bubbles */
    .chat-bubble {
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.5s ease-in-out;
        max-width: 85%;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        align-self: flex-end;
        color: white;
        border-bottom-right-radius: 2px;
        margin-left: auto;
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        align-self: flex-start;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom-left-radius: 2px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# -------------------------------------------------------------
# LOGIQUE CHROMADB & RAG
# -------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

# Initialisation ChromaDB (persistant)
CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Sélection ou création de la collection
collection = chroma_client.get_or_create_collection(
    name="my_documents", 
    embedding_function=embedding_function
)

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, size=800, overlap=100):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

def get_answer(question, chat_history, top_k=4, temperature=0.7):
    # 1. Recherche de similarité dans ChromaDB
    results = collection.query(query_texts=[question], n_results=top_k)
    context = "\n\n".join(results['documents'][0])
    
    # 2. Construction du prompt avec instructions strictes
    messages = [
        {"role": "system", "content": f"""Tu es VaultAI, un assistant RAG expert. 
        Tes instructions sont STRICTES :
        1. Réponds UNIQUEMENT en utilisant le contexte fourni ci-dessous.
        2. Si la réponse n'est pas dans le contexte, dis explicitement : "Désolé, je ne trouve pas cette information dans vos documents."
        3. Ne fais pas appel à tes connaissances générales.
        4. Cite le nom du document source si possible.
        """}
    ]
    
    # Ajouter l'historique récent
    for msg in chat_history[-5:]:
        messages.append(msg)
        
    messages.append({
        "role": "user", 
        "content": f"CONTEXTE:\n{context}\n\nQUESTION: {question}"
    })
    
    try:
        # 3. Génération via LLM avec paramètres réglables
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content, results['metadatas'][0]
    except Exception as e:
        return f"Erreur Groq: {e}", []

# -------------------------------------------------------------
# INTERFACE STREAMLIT
# -------------------------------------------------------------
st.title("🤖 VaultAI - Intelligent RAG")

# Sidebar
with st.sidebar:
    st.header("⚙️ Réglages & Paramètres")
    
    st.subheader("🤖 Intelligence")
    temp = st.slider(
        label="Niveau de créativité (Température)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="0.0 = Factuel | 1.0 = Créatif"
    )
    
    top_k = st.slider(
        label="Profondeur de recherche (Top-K)", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Nombre de morceaux lus par l'IA."
    )
    
    st.divider()
    st.subheader("📝 Découpage (Chunking)")
    chunk_size = st.number_input(
        label="Taille des morceaux", 
        min_value=100, 
        max_value=2000, 
        value=800
    )
    
    chunk_overlap = st.number_input(
        label="Chevauchement (Overlap)", 
        min_value=0, 
        max_value=500, 
        value=100
    )

    st.divider()
    st.header("📂 Gestion des Documents")
    uploaded_files = st.file_uploader("Indexez vos PDFs", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("✨ Indexer")
    with col2:
        clear_btn = st.button("🗑️ Vider")

    if build_btn:
        if uploaded_files:
            with st.status("🛠️ Pipeline RAG en cours...", expanded=True) as status:
                st.write("1. 📄 Extraction du texte...")
                for f in uploaded_files:
                    text = extract_text_from_pdf(f)
                    st.write(f"   - {f.name} extrait.")
                    
                    st.write("2. ✂️ Création des chunks...")
                    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
                    
                    st.write("3. 🧠 Vectorisation & Stockage ChromaDB...")
                    ids = [f"{f.name}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": f.name} for _ in chunks]
                    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
                status.update(label="✔ Base RAG construite !", state="complete", expanded=False)
                st.success(f"{len(uploaded_files)} document(s) indexés.")
        else:
            st.error("Veuillez uploader un fichier.")
            
    if clear_btn:
        ids = collection.get()['ids']
        if ids:
            collection.delete(ids=ids)
            st.success("Base de données vidée.")
        else:
            st.info("La base est déjà vide.")

    # Aperçu de la base
    st.divider()
    st.header("📊 Dossier Vectoriel")
    count = collection.count()
    st.metric("Vecteurs actifs", count)
    if count > 0:
        with st.expander("Sources indexées"):
            sources = list(set([m['source'] for m in collection.get()['metadatas']]))
            for s in sources:
                st.write(f"- {s}")

    st.divider()
    st.write("Statut: **Prêt**" if GROQ_API_KEY else "Statut: ⚠️ **Clé API manquante**")

# Chat Interface
if "messages_premium" not in st.session_state:
    st.session_state.messages_premium = []

for msg in st.session_state.messages_premium:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Posez votre question...")

if user_input:
    st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Recherche de similarité & Génération..."):
        answer, sources = get_answer(user_input, st.session_state.messages_premium, top_k=top_k, temperature=temp)
        
        st.session_state.messages_premium.append({"role": "user", "content": user_input})
        st.session_state.messages_premium.append({"role": "assistant", "content": answer})
        
        st.markdown(f'<div class="chat-bubble assistant-bubble">{answer}</div>', unsafe_allow_html=True)
        
        if sources:
            source_names = list(set([s['source'] for s in sources]))
            st.caption(f"Pipeline RAG : {len(sources)} vecteurs similaires trouvés dans {', '.join(source_names)}")
            
    st.rerun()
