"""
Treina um modelo YOLO customizado para detecção de celulares no domínio
escolar, partindo de pesos pré-treinados no COCO (transfer learning).

Uso:
    python model/training/train.py \
        --data model/datasets/school_phones/data.yaml \
        --epochs 100 \
        --base-model yolov8n.pt \
        --output-name custom_school_phone_v1

O checkpoint final treinado (best.pt) deve ser renomeado/copiado para um
caminho contendo "custom" no nome (ex.: custom_school_phone_v1.pt) e
referenciado em MODEL_PATH no .env — a factory em app/vision/detector.py
usa essa convenção para escolher CustomSchoolPhoneDetector automaticamente.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train(data_yaml: Path, epochs: int, base_model: str, output_name: str, image_size: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        name=output_name,
        # Data augmentation relevante para o domínio: variações de
        # iluminação, escala e oclusão parcial ajudam a generalizar para
        # celulares em diferentes ângulos/distâncias/posições.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.4,
        fliplr=0.5,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    final_path = Path("model") / f"{output_name}.pt"
    shutil.copy2(best_weights, final_path)
    print(f"Modelo treinado salvo em: {final_path}")
    print("Renomeie/mantenha 'custom' no nome do arquivo para que a factory de detectores o reconheça automaticamente.")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--output-name", default="custom_school_phone")
    parser.add_argument("--image-size", default=640, type=int)
    args = parser.parse_args()

    train(args.data, args.epochs, args.base_model, args.output_name, args.image_size)


if __name__ == "__main__":
    main()
