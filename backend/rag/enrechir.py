import json
import subprocess

INPUT_FILE = "dataset/plantuml/plantuml_dataset.json"
OUTPUT_FILE = "plantuml_dataset_enriched.json"
with open ()

def ollama_generate(prompt: str, model: str = "llama3") -> str:
    """Appelle Ollama en ligne de commande"""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8").strip()

def generate_description_with_llm(code: str) -> str:
    prompt = f"""
    Tu es un expert UML.
    Analyse ce code PlantUML et donne une brève description (1-2 phrases maximum) du diagramme :

    {code}
    """
    return ollama_generate(prompt)

def enrich_dataset_with_llm(data: list) -> list:
    enriched = []
    for entry in data:
        code = entry.get("code", "")
        entry["description"] = generate_description_with_llm(code)
        enriched.append(entry)
    return enriched

if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched_data = enrich_dataset_with_llm(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset enrichi avec descriptions générées par Ollama -> {OUTPUT_FILE}")
