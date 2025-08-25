from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
INDEX_DIR = "storage/vector_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Charger l'index FAISS local ---
vector_store = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)
total_docs = len(vector_store.docstore._dict)
print(f"Total de documents dans l'index FAISS : {total_docs}")

# --- Liste de requêtes de test ---
test_queries = [
    "diagramme de contexte application e-commerce"
]

# --- Fonction de validation ---
def test_query(query, top_k=3, keywords=None):
    print(f"\n=== Test de la requête : '{query}' ===")
    results = vector_store.similarity_search(query, k=top_k)
    
    if not results:
        print("Aucun résultat trouvé.")
        return
    
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print("Tags :", r.metadata.get("tags", "N/A"))
        print("Source :", r.metadata.get("source", "N/A"))
        print("Description (extrait) :", r.page_content[:300], "...\n")

# --- Boucle sur toutes les requêtes de test ---
for query in test_queries:
    test_query(query)
