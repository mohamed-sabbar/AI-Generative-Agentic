from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# Initialisation du LLM Ollama (version mise à jour)
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# Créer un message humain (HumanMessage)
messages = [HumanMessage(content="Fais-moi un diagramme UML de class de gestion de stock avec plantuml")]

# Génération de la réponse
response = llm.invoke(messages)

# Affichage du résultat
print("=== Diagramme UML généré ===")
print(response.content)
