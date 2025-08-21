from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import json
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
prompt ="Fais-moi un diagramme UML de class de gestion de stock avec plantuml"

    
messages = [HumanMessage(content=prompt)]

    
response = llm.invoke(messages)

print(response.content)



