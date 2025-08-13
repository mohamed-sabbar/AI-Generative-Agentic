from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "storage/vector_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


query = "sequence diagram login process"
results = vector_store.similarity_search(query, k=3)

print("Résultats trouvés :")
for r in results:
    print(r.metadata, "\n", r.page_content[:300], "...\n")
