from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import os
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chargement de l'index FAISS


def load_vector(diagram_type:str):
     index_path = os.path.join("storage", f"index_{diagram_type}")
     return FAISS.load_local(
    index_path,
    embeddings,
    allow_dangerous_deserialization=True
)
# Récupérateur
retriever = load_vector("uml").as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM Ollama
llm = ChatOllama(
    model="llama3.1",       # Modèle Ollama installé localement
    temperature=0
)

# Chaîne RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Question
query = "diagramme de usecase de gestion de stock avec plantuml"
result = rag_chain.invoke({"query": query})

print(result['result'])
