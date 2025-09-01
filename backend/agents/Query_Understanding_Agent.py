from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import json
import re
# -------------------------
# 1. Définir la structure attendue
# -------------------------
class QueryStructure(BaseModel):
    diagram_type: str = Field(..., description="Type de diagramme (uml, c4, mermaid, mindmap, etc.)")
    subject: str = Field(..., description="Le sujet principal du diagramme (ex: SelectionManager, UserService)")
    constraints: str = Field(..., description="Contraintes ou précisions demandées par l’utilisateur")

# -------------------------
# 2. LLM (Ollama local)
# -------------------------
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# -------------------------
# 3. Prompt pour analyser la requête
# -------------------------
prompt = ChatPromptTemplate.from_template("""
Tu es un agent qui analyse des requêtes utilisateur.
La requête sera une question demandant un diagramme (UML, C4, etc.).

Tâche : identifie
- le type de diagramme
- le sujet principal
- les contraintes

Réponds uniquement au format JSON suivant :
{{
  "diagram_type": "...",
  "subject": "...",
  "constraints": "..."
}}

Requête utilisateur : "{user_query}"
""")

# -------------------------
# 4. Fonction Agent
# -------------------------
def query_understading_agent(user_query: str) -> QueryStructure:
    final_prompt = prompt.format_messages(user_query=user_query)
    response = llm.invoke(final_prompt)
    content = response.content

    # Extraire le JSON entre accolades
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError(f"Impossible de trouver le JSON dans la réponse : {content}")

    data = json.loads(match.group())
    return QueryStructure(**data)

# -------------------------
# 5. Test
# -------------------------
if __name__ == "__main__":
    query = "diagramme de classe SelectionManager avec plantuml"
    result = query_understading_agent(query)
    print(result.model_dump())
