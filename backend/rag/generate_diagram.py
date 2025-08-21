from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chargement de l'index FAISS
vector_store = FAISS.load_local(
    "storage/vector_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Récupérateur
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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
query = "diagramme de classe SelectionManager avec plantuml"
result = rag_chain.invoke({"query": query})

print(result['result'])
