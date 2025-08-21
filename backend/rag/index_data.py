import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings   
from langchain.docstore.document import Document
from dotenv import load_dotenv
import json
load_dotenv() 

DATA_DIR = "dataset/plantuml/dataset.json"  
INDEX_DIR = "storage/vector_index"


with open(DATA_DIR,"r",encoding='utf-8') as f:
      data=json.load(f)
documents = []
for entry in data:
    # Vérifie si "code" est un dict ou une string
    code = entry.get("code", "")
    if isinstance(code, dict):
        # Prend le champ "content" si c'est un dict
        content = code.get("content", "")
    else:
        content = str(code)  # assure une string

    metadata = {
        "prompt": entry.get("prompt", ""),
        "langage": entry.get("langage", ""),
        "type": entry.get("type", ""),
        "tags": entry.get("tags", []),
        "description": entry.get("description", ""),
        "source": entry.get("source", "")
    }

    documents.append(Document(page_content=content, metadata=metadata))

print(f"Nombre de documents chargés : {len(documents)}")
print(documents[19])
print(len(data))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Créer l'index FAISS ---
vector_store = FAISS.from_documents(documents, embeddings)

# --- Sauvegarder l'index localement ---
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)
vector_store.save_local(INDEX_DIR)
print("Indexation terminée !")

# --- Recharger l'index pour tester ---
vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- Test d'une requête ---
query = "diagramme de classe SelectionManager"
results = vector_store.similarity_search(query, k=3)

print("\nRésultats trouvés :")
for i, r in enumerate(results):
    print(f"\nResult {i+1}:")
    print("Tags :", r.metadata.get("tags", "N/A"))
    print("Source :", r.metadata.get("source", "N/A"))
    print("Description :", r.metadata.get("description", "N/A"))
    print("Extrait du code :", r.page_content[:300], "...\n")

# --- Vérifier le nombre total de documents indexés ---
print("Total de documents dans l'index FAISS :", vector_store.index.ntotal)
