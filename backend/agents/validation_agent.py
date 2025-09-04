import requests
from Query_Understanding_Agent import query_understading_agent
from Retrieval_Agent import retrieval_agent
from Reranker_Agent import reranker_agent
from Generation_Agent import generation_agent
def validate_and_generate_diagram(diagram_code: str, diagram_type: str = "plantuml", output_file: str = "diagram.png", format: str = "png") -> bool:
    """
    Valide un diagramme et génère l'image via l'API Kroki.
    Retourne True si le rendu est correct, False sinon.
    """
    url = f"https://kroki.io/{diagram_type}/{format}"
    headers = {"Content-Type": "text/plain"}

    try:
        response = requests.post(url, data=diagram_code.encode("utf-8"), headers=headers)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return True
        else:
            print("❌ Erreur Kroki :", response.status_code, response.text)
            return False
    except Exception as e:
        print("❌ Exception :", e)
        return False
if __name__=="__main__":
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
    is_valid=validate_and_generate_diagram(diagram_code=diagram_code)
    print(is_valid)
