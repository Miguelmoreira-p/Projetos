"""
Prepara um dataset no formato YOLO (imagens + labels .txt) a partir de
fontes públicas/licenciadas, para treinar um detector de celulares
especializado no domínio escolar.

NÃO use imagens reais de estudantes. Fontes recomendadas para começar:

- COCO (classe "cell phone"): https://cocodataset.org/ — licença CC BY 4.0.
  Já contém milhares de exemplos de celulares em diversos contextos, ângulos
  e distâncias.
- Open Images V7 (classe "Mobile phone"):
  https://storage.googleapis.com/openimages/web/index.html — licença
  CC BY 4.0. Complementa o COCO com mais variedade de ambientes.
- Datasets sintéticos/próprios: fotos tiradas em ambiente controlado
  (voluntários adultos, sem menores de idade, com consentimento explícito),
  variando modelo de celular, iluminação, ângulo, distância e oclusão
  parcial (celular na mão, sobre a mesa, parcialmente coberto).

Este script assume que os dados brutos já foram baixados manualmente (por
questões de licença, não fazemos download automático) e organiza/valida a
estrutura esperada pelo Ultralytics:

    model/datasets/school_phones/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
        data.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

CLASS_NAMES = ["cell_phone"]


def validate_pair(image_path: Path, label_path: Path) -> bool:
    if not label_path.exists():
        print(f"[aviso] Sem label para {image_path.name}, pulando.")
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[erro] Linha de label malformada em {label_path.name}: {line!r}")
                return False
            class_id = int(parts[0])
            if class_id >= len(CLASS_NAMES):
                print(f"[erro] class_id {class_id} fora do intervalo em {label_path.name}")
                return False
    return True


def build_dataset(raw_images_dir: Path, raw_labels_dir: Path, output_dir: Path, val_split: float = 0.2) -> None:
    images = sorted([p for p in raw_images_dir.glob("*.jpg")] + [p for p in raw_images_dir.glob("*.png")])
    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {raw_images_dir}")

    valid_pairs = []
    for image_path in images:
        label_path = raw_labels_dir / f"{image_path.stem}.txt"
        if validate_pair(image_path, label_path):
            valid_pairs.append((image_path, label_path))

    split_index = int(len(valid_pairs) * (1 - val_split))
    splits = {"train": valid_pairs[:split_index], "val": valid_pairs[split_index:]}

    for split_name, pairs in splits.items():
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in pairs:
            shutil.copy2(image_path, images_out / image_path.name)
            shutil.copy2(label_path, labels_out / f"{image_path.stem}.txt")

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Dataset preparado em {output_dir}")
    print(f"  train: {len(splits['train'])} imagens")
    print(f"  val:   {len(splits['val'])} imagens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--raw-labels", required=True, type=Path)
    parser.add_argument("--output", default=Path("model/datasets/school_phones"), type=Path)
    parser.add_argument("--val-split", default=0.2, type=float)
    args = parser.parse_args()

    build_dataset(args.raw_images, args.raw_labels, args.output, args.val_split)


if __name__ == "__main__":
    main()
