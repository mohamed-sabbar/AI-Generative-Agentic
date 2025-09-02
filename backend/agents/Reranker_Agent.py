# reranker_agent.py

from langchain_ollama import ChatOllama
from typing import List, Tuple

# -------------------------
# 1. LLM pour le reranking
# -------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# -------------------------
# 2. Définir le Reranker Agent
# -------------------------
def reranker_agent(user_query: str, documents: List[str]) -> str:
    """
    Classe/rerank les documents récupérés et retourne le meilleur.
    """
    # Construire le prompt pour le LLM
    prompt = f"""
Tu es un agent qui choisit le document le plus pertinent pour une requête utilisateur.
Requête : "{user_query}"

Voici les documents récupérés : 
{chr(10).join([f"Doc{i+1}: {doc[:300]}..." for i, doc in enumerate(documents)])}

Tâche : Retourne **uniquement le numéro** du document le plus pertinent (ex: Doc1, Doc2, ...).
"""

    # Appel LLM
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Extraire le numéro du document
    import re
    match = re.search(r'Doc(\d+)', content)
    if not match:
        # Si LLM ne donne pas de numéro clair, retourner le premier document
        return documents[0]
    
    doc_index = int(match.group(1)) - 1
    return documents[doc_index]

# -------------------------
# 3. Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    from Retrieval_Agent import retrieval_agent
    from Query_Understanding_Agent import query_understading_agent

    user_query = "diagramme de classe d'une application qui liste les examens des cours avec PlantUML"
    query_struct = query_understading_agent(user_query)

    # Récupération des documents
    docs = retrieval_agent(query_struct)
    
    # Reranking
    best_doc = reranker_agent(user_query, docs)
    print("Meilleur document sélectionné :\n", best_doc)
