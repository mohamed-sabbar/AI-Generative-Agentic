from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os
import re

# -------------------------
# 1. Chargement des embeddings et de l'index FAISS
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_index(diagram_type: str):
    """
    Charge l'index correspondant au type de diagramme
    """
    current_dir = os.path.dirname(__file__)  # dossier du script (agent/)
    project_root = os.path.dirname(current_dir)  # remonte à la racine
    index_path = os.path.join(project_root, "rag", "storage", f"index_{diagram_type}")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# -------------------------
# 2. Définir le Retrieval Agent
# -------------------------
def retrieval_agent(query_structure) -> List[str]:
    """
    Récupère les documents pertinents à partir de la sortie du Query Understanding Agent
    """
    # Charger le vector store correspondant au type de diagramme
    match = re.match(r"(\w+)", query_structure.diagram_type.lower())
    if match:
     diagram_type_clean = match.group(1)
     
    vector_store = load_index(diagram_type_clean)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Construire la requête à partir de la structure de la requête
    search_query = f"{query_structure.diagram_type} {query_structure.subject} {query_structure.constraints}"

    # Récupérer les documents
    docs = retriever.get_relevant_documents(search_query)

    # Retourner le contenu des documents
    return [doc.page_content for doc in docs]

# -------------------------
# 3. Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    from Query_Understanding_Agent import query_understading_agent

    user_query = "diagramme de classe SelectionManager avec PlantUML"
    query_struct = query_understading_agent(user_query)

    results = retrieval_agent(query_struct)

    print("Documents pertinents :")
    for i, doc in enumerate(results, 1):
        print(f"Doc{i} :", doc, "\n")  # affiche seulement les 500 premiers caractères
