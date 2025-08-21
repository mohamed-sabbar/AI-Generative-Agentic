from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import json
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
prompt ="diagramme de classe SelectionManager avec plantuml"

    
messages = [HumanMessage(content=prompt)]

    
response = llm.invoke(messages)

print(response.content)



