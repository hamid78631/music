import os

folders = [
    "data/raw",
    "data/cleaned",
    "notebooks",
    "src",
    "app"
]

files = [
    "README.md",
    "requirements.txt",
    "src/data_cleaning.py",
    "src/pca_analysis_module.py",
    "src/clustering.py",
    "src/recommendation.py",
    "app/main.py",
    "notebooks/exploration.ipynb"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        f.write("")  # crée un fichier vide

print("Structure du projet Aromind créée !")
