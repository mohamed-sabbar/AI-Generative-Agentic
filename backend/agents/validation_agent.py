# validation_agent.py

import subprocess
import tempfile
import os

# Chemin vers le plantuml.jar sur ton système Windows
PLANTUML_JAR_PATH = r"C:\plantuml\plantuml.jar"

def validate_plantuml(diagram_code: str) -> bool:
    """
    Valide un diagramme PlantUML en essayant de le rendre.
    Retourne True si le code est correct, False sinon.
    """
    tmp_file_path = None
    try:
        # Crée un fichier temporaire .puml
        with tempfile.NamedTemporaryFile(delete=False, suffix=".puml") as tmp_file:
            tmp_file.write(diagram_code.encode("utf-8"))
            tmp_file_path = tmp_file.name

        # Appelle PlantUML via Java pour générer un PNG
        subprocess.run(
            ["java", "-jar", PLANTUML_JAR_PATH, "-tpng", tmp_file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print("Erreur PlantUML :", e.stderr.decode())
        return False
    finally:
        # Supprime le fichier temporaire
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# -------------------------
# Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    from Generation_Agent import generation_agent
    from Query_Understanding_Agent import query_understading_agent
    from Retrieval_Agent import retrieval_agent
    from Reranker_Agent import reranker_agent

    user_query = "diagramme de classe d'une application qui liste les examens des cours avec plantuml"

    query_struct = query_understading_agent(user_query)
    docs = retrieval_agent(query_struct)
    best_doc = reranker_agent(user_query, docs)
    diagram_code = generation_agent(user_query, best_doc)

    print("Code généré :\n", diagram_code)

    is_valid = validate_plantuml(diagram_code)
    if is_valid:
        print("✅ Le diagramme est valide !")
    else:
        print("❌ Le diagramme contient des erreurs.")
