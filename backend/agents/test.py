import re

diagram_type = "uml class diagram"

# Extraire le premier mot (avant un espace)
match = re.match(r"(\w+)", diagram_type.lower())

if match:
    diagram_type_clean = match.group(1)
    print(diagram_type_clean)  # affiche "uml"