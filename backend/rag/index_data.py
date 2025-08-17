import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings   
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv() 

DATA_DIR = "dataset/plantuml/"  
INDEX_DIR = "storage/vector_index"

documents = []
for filename in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
            code = f.read()
        metadata = {
            "type": "class diagram",
            "filename": filename,
            "tool": "PlantUML"
        }
        documents.append(Document(page_content=code, metadata=metadata))



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 3️⃣ Créer la base FAISS et indexer
vector_store = FAISS.from_documents(documents, embeddings)

# 4️⃣ Sauvegarder l'index
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)
vector_store.save_local(INDEX_DIR)

print("Indexation terminée !")
# Charger l'index
vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Faire une requête
query = "class diagram login process"
results = vector_store.similarity_search(query, k=3)  # k = nombre de résultats

print("Résultats trouvés :")
for r in results:
    print(r.metadata, "\n", r.page_content[:300], "...")  # afficher un extrait
