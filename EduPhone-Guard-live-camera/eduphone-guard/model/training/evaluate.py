"""
Avalia um modelo treinado, reportando não apenas precision/recall/mAP
(métricas padrão do Ultralytics), mas também a False Positive Rate (FPR)
de forma explícita — porque, neste sistema, gerar alertas falsos é um
problema tão importante quanto deixar de detectar um celular real.

Uso:
    python model/training/evaluate.py --weights model/custom_school_phone.pt --data model/datasets/school_phones/data.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(weights_path: Path, data_yaml: Path, confidence: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), conf=confidence)

    # `metrics.box` expõe precision/recall/mAP por classe (Ultralytics >= 8.x).
    precision = float(metrics.box.mp)  # mean precision
    recall = float(metrics.box.mr)  # mean recall
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Ultralytics não expõe FPR diretamente; aproximamos usando a matriz de
    # confusão (última linha/coluna = background). FP = predições marcadas
    # como "objeto" que na verdade eram background.
    confusion_matrix = metrics.confusion_matrix.matrix
    false_positives = float(confusion_matrix[:-1, -1].sum())
    true_negatives_proxy = float(confusion_matrix[-1, -1])
    denom = false_positives + true_negatives_proxy
    false_positive_rate = false_positives / denom if denom > 0 else 0.0

    report = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "false_positives_estimate": false_positives,
        "false_positive_rate_estimate": false_positive_rate,
    }

    print(json.dumps(report, indent=2))
    print(
        "\nAtenção: false_positive_rate_estimate é uma aproximação derivada da "
        "matriz de confusão de validação. Para uma medição mais confiável em "
        "produção, monitore a proporção de eventos marcados como "
        "'falso positivo' pelos revisores humanos ao longo do tempo."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--confidence", default=0.5, type=float)
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.confidence)


if __name__ == "__main__":
    main()
