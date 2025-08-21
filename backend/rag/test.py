from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
import json

INPUT_FILE = "dataset/plantuml/plantuml_dataset.json"
OUTPUT_FILE = "plantuml_dataset_enriched.json"

# Charger le dataset JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialisation du modèle Ollama
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

results = []

for i in range(len(data)):
    prompt = f"""
Tu es un expert UML. 
Analyse le code PlantUML ci-dessous et renvoie un objet JSON STRICT avec les champs suivants :
- prompt (reprend le même prompt donné)
- code (reprend le code donné)
- langage (reprend la même valeur donnée)
- type (type du diagramme UML, ex: "sequence", "usecase", "class", "component", etc.)
- tags (liste de mots-clés pertinents extraits du code)
- description (1-2 phrases max décrivant le diagramme, sans intro ni formatage)
- source (reprend la même valeur donnée)

IMPORTANT : Réponds uniquement avec du JSON valide, pas de texte hors JSON.

Code à analyser :
{data[i]["code"]}
"""

    # Construire le message correctement
    messages = [HumanMessage(content=prompt)]

    # Exécuter l'appel au LLM
    response = llm.invoke(messages)

    try:
        enriched = json.loads(response.content) 
    except json.JSONDecodeError:
        print(f"⚠️ Erreur de parsing JSON à l’index {i}")
        print(response.content)

# Sauvegarde dans un fichier enrichi
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Enrichissement terminé. Résultats enregistrés dans {OUTPUT_FILE}")
