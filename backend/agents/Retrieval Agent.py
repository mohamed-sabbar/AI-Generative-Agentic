import os
import Query_Understanding_Agent
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def load_index(diagram_type:str):
    index_path = os.path.join("storage", f"index_{diagram_type}")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
def retrieval_agent():
    pass
if __name__=="__main__":
    pass