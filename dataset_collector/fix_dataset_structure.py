from pathlib import Path
import shutil

ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"

print("Verificando arquivos com barras invertidas no nome...\n")

moved = 0
errors = 0

for f in list(DATASET_DIR.glob("*")):
    if "\\" not in f.name and "/" not in f.name:
        continue  # arquivo normal, pular

    # Normalizar o caminho: "images\train\arquivo.jpg" → dest correto
    norm = f.name.replace("\\", "/")
    dest = DATASET_DIR / norm
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(f), str(dest))
        moved += 1
        if moved % 5000 == 0:
            print(f"  {moved} arquivos movidos...")
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  [ERRO] {f.name}: {e}")

print(f"\nConcluído: {moved} arquivos movidos, {errors} erros.")

# Verificar resultado
for split in ["train", "val", "test"]:
    for tipo in ["images", "labels"]:
        p = DATASET_DIR / tipo / split
        if p.exists():
            ext = "*.jpg" if tipo == "images" else "*.txt"
            n = len(list(p.glob(ext)))
            print(f"  dataset/{tipo}/{split}: {n} arquivos")
