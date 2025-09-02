# generation_agent.py

from langchain_ollama import ChatOllama
from Query_Understanding_Agent import query_understading_agent
from Retrieval_Agent import retrieval_agent
from Reranker_Agent import reranker_agent
# -------------------------
# 1. Initialiser le LLM
# -------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# -------------------------
# 2. Définir le Generation Agent
# -------------------------
def generation_agent(user_query: str, best_doc: str) -> str:
    """
    Génère le code final du diagramme à partir de la requête et du document sélectionné.
    """
    prompt = f"""
Tu es un agent spécialisé en génération de diagrammes (PlantUML, Mermaid, C4, etc.).

Requête utilisateur :
{user_query}

Document pertinent sélectionné :
{best_doc}

Tâche :
- Génère uniquement le code complet du diagramme correspondant.
- Respecte la syntaxe exacte du type demandé (PlantUML, Mermaid, etc.).
- Ne retourne rien d'autre que le code du diagramme.
"""

    response = llm.invoke(prompt)
    return response.content.strip()

# -------------------------
# 3. Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
 

    # Requête utilisateur
    user_query = "diagramme de classe d'une application qui liste les examens  de  avec PlantUML"

    # Étape 1 : Compréhension de la requête
    query_struct = query_understading_agent(user_query)

    # Étape 2 : Récupération de documents
    docs = retrieval_agent(query_struct)

    # Étape 3 : Reranking
    best_doc = reranker_agent(user_query, docs)

    # Étape 4 : Génération du diagramme final
    diagram_code = generation_agent(user_query, best_doc)

    print("Diagramme généré :\n")
    print(diagram_code)
