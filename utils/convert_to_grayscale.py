#!/usr/bin/env python3
"""
convert_to_grayscale.py — Converte todas as imagens do dataset para P&B.

Converte cada .jpg de dataset/images/train/ e dataset/images/val/ para
escala de cinza (3 canais BGR iguais), sobrescrevendo os arquivos originais.
Os labels (.txt) não são alterados — a bbox não muda com a conversão.

Uso:
    python convert_to_grayscale.py           # converte de verdade
    python convert_to_grayscale.py --dry-run # só mostra o que faria
    python convert_to_grayscale.py --split train  # só train
"""

import cv2
import argparse
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"


def convert_split(split: str, dry_run: bool):
    img_dir = DATASET_ROOT / "images" / split
    if not img_dir.exists():
        print(f"  [AVISO] {img_dir} não encontrada, pulando.")
        return 0

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not imgs:
        print(f"  [AVISO] Nenhuma imagem em {img_dir}")
        return 0

    print(f"  {split}: {len(imgs)} imagens")

    converted = 0
    for i, p in enumerate(imgs):
        if dry_run:
            converted += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(p), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        converted += 1
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(imgs)} convertidas...")

    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]

    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{'='*50}")
    print(f"  {tag}CONVERSÃO PARA ESCALA DE CINZA")
    print(f"{'='*50}\n")

    total = 0
    for split in splits:
        total += convert_split(split, args.dry_run)

    print(f"\n  {tag}Total convertido: {total} imagens")
    if not args.dry_run:
        print("  Labels (.txt) não foram alterados.")
        print("\n  Próximo passo: python training/train.py")


if __name__ == "__main__":
    main()
