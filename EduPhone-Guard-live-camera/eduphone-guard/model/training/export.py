"""
Exporta um modelo treinado para um formato adequado à inferência em
produção (ONNX, por padrão — mais portável entre ambientes que o formato
nativo do PyTorch).

Uso:
    python model/training/export.py --weights model/custom_school_phone.pt --format onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path


def export(weights_path: Path, export_format: str, image_size: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported_path = model.export(format=export_format, imgsz=image_size)
    print(f"Modelo exportado para: {exported_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "openvino"])
    parser.add_argument("--image-size", default=640, type=int)
    args = parser.parse_args()

    export(args.weights, args.format, args.image_size)


if __name__ == "__main__":
    main()
