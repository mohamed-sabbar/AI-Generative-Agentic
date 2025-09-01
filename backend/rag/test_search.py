from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama
# --- Config ---
INDEX_DIR = "storage/index_uml"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Charger FAISS
vector_store = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

# LLM (ici OpenAI, mais tu peux remplacer par autre)
llm = ChatOllama(
    model="llama3.1",       # Modèle Ollama installé localement
    temperature=0
)

def retrieve_and_rerank(query, top_k=5):
    # 1. Récupération FAISS
    candidates = vector_store.similarity_search(query, k=top_k)

    # 2. Construire un prompt de reranking
    context_text = "\n\n".join(
        [f"Document {i+1}:\n{c.page_content[:500]}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
Tu es un assistant qui aide à trouver le meilleur diagramme UML pour une requête utilisateur.

Requête utilisateur :
\"\"\"{query}\"\"\"

Voici les {top_k} documents candidats :
{context_text}

Analyse les documents et indique :
1. Le numéro du document qui correspond le mieux à la requête.
2. Une brève justification.

Réponds sous format JSON :
{{
  "best_document": <numéro>,
  "reason": "<raison>"
}}
    """

    # 3. LLM choisit le meilleur
    response = llm.invoke(prompt)
    return response.content

# Exemple d’utilisation
query = "donne moi diagramme UML d'une application qui liste les examens des cours"
result = retrieve_and_rerank(query)
print(result)
