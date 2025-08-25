import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Charger les variables d'environnement
load_dotenv()

# --- Configuration ---
DATASET_PATHS = {
    "uml": "dataset/UML/dataset_UML.json",
    "c4": "dataset/C4/plantuml_c4_dataset.json",
    "mindmap": "dataset/mindmap/mindmap_dataset.json"
}
INDEX_DIR = "storage"

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_dataset(file_path: str):
    """Charge un dataset JSON et retourne une liste d'entrées."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_to_str(value):
    """Transforme n'importe quelle valeur en string propre."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_documents(data):
    """Transforme les données JSON en documents LangChain."""
    documents = []
    for entry in data:
        # Contenu principal
        code = entry.get("code", "")
        content = safe_to_str(code)

        metadata = {
            "prompt": safe_to_str(entry.get("prompt", "")),
            "langage": safe_to_str(entry.get("langage", "")),
            "type": safe_to_str(entry.get("type", "")),
            "tags": safe_to_str(entry.get("tags", [])),
            "description": safe_to_str(entry.get("description", "")),
            "source": safe_to_str(entry.get("source", "")),
        }

        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def index_dataset(dataset_name, dataset_path):
    """Construit un index FAISS pour un dataset donné."""
    print(f"\n📂 Indexation du dataset: {dataset_name}")
    data = load_dataset(dataset_path)
    documents = build_documents(data)

    # Construire l’index
    vector_store = FAISS.from_documents(documents, embeddings)

    # Sauvegarder
    index_path = os.path.join(INDEX_DIR, f"index_{dataset_name}")
    os.makedirs(INDEX_DIR, exist_ok=True)
    vector_store.save_local(index_path)

    print(f"✅ Index sauvegardé : {index_path} ({len(documents)} documents)")


if __name__ == "__main__":
    for name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            try:
                index_dataset(name, path)
            except Exception as e:
                print(f"❌ Erreur lors de l'indexation de {name} : {e}")
        else:
            print(f"⚠️ Dataset introuvable : {path}")
