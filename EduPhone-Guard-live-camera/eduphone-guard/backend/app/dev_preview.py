"""
Script de teste rápido e visual: roda o detector sobre um vídeo e grava um
NOVO vídeo com as bounding boxes e a confiança desenhadas em cada frame
onde um celular foi detectado.

Não depende de Supabase, banco de dados, ou de nenhuma outra parte do
sistema — serve só para você confirmar visualmente se o modelo está
detectando (ou não) o celular no seu vídeo, antes de se preocupar com o
resto do pipeline (persistência, eventos, dashboard etc.).

Uso:
    python -m app.dev_preview --video caminho/para/seu_video.mp4

Gera por padrão: data/clips/preview_annotated.mp4

Opcional:
    --confidence 0.5     (limiar de confiança só para esta pré-visualização;
                           por padrão usa o mesmo valor de MIN_CONFIDENCE do .env)
    --output caminho.mp4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from app.config import get_settings
from app.vision.detector import build_detector

logging.basicConfig(level="INFO")
logger = logging.getLogger("eduphone_guard.dev_preview")


def run_preview(video_path: str, output_path: str, confidence: float) -> dict:
    settings = get_settings()

    detector = build_detector(
        model_path=settings.model_path,
        min_confidence=confidence,
        inference_resize_width=settings.inference_resize_width,
    )
    logger.info("Usando detector: %s (confiança mínima=%.2f)", detector.name, confidence)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"Não foi possível abrir o vídeo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_number = 0
    frames_with_detection = 0
    max_confidence_seen = 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detector.detect(frame, frame_number, camera_id="preview")

        if detections:
            frames_with_detection += 1

        for det in detections:
            max_confidence_seen = max(max_confidence_seen, det.confidence)
            x1, y1, x2, y2 = (int(v) for v in det.bbox.as_tuple())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"celular {det.confidence:.0%}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)
        frame_number += 1

    capture.release()
    writer.release()

    summary = {
        "total_frames": frame_number,
        "frames_with_detection": frames_with_detection,
        "max_confidence_seen": round(max_confidence_seen, 3),
        "output_path": str(output_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Caminho para o vídeo de entrada (mp4).")
    parser.add_argument("--output", default="data/clips/preview_annotated.mp4")
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Limiar de confiança para esta pré-visualização (padrão: MIN_CONFIDENCE do .env).",
    )
    args = parser.parse_args()

    settings = get_settings()
    confidence = args.confidence if args.confidence is not None else settings.min_confidence

    summary = run_preview(args.video, args.output, confidence)

    print("\n=== Resultado da pré-visualização ===")
    print(f"Total de frames processados: {summary['total_frames']}")
    print(f"Frames com celular detectado: {summary['frames_with_detection']}")
    print(f"Confiança máxima observada: {summary['max_confidence_seen']}")
    print(f"Vídeo anotado salvo em: {summary['output_path']}")

    if summary["frames_with_detection"] == 0:
        print(
            "\nNenhuma detecção encontrada. Possíveis causas: o celular está "
            "pequeno/distante demais, ângulo incomum, iluminação ruim, ou o "
            "limiar de confiança está alto demais para este vídeo. Tente "
            "rodar novamente com --confidence 0.3 para ver se aparece algo "
            "(sinal de que o problema é calibração, não detecção zero)."
        )


if __name__ == "__main__":
    main()
