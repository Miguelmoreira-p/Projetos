"""
Interface abstrata de detecção de objetos + implementações concretas.

Design intencional: todo o resto do sistema (tracker, event_detector, API)
depende apenas da interface `Detector`. Isso permite trocar o modelo
(genérico pré-treinado -> modelo customizado treinado para o domínio escolar)
sem alterar qualquer outro componente.

NENHUMA implementação aqui deve realizar reconhecimento facial ou qualquer
tentativa de identificação de pessoas. O único objeto de interesse é o
celular ("cell phone" / "cell_phone").
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from app.schemas.detection import BoundingBox, Detection

logger = logging.getLogger("eduphone_guard.vision")

# Nomes de classe do COCO (usado pelos pesos padrão do YOLOv8) que mapeiam
# para "celular". O dataset COCO usa "cell phone".
COCO_PHONE_CLASS_NAMES = {"cell phone", "mobile phone"}


class Detector(ABC):
    """Contrato que qualquer detector de objetos deve implementar."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_number: int, camera_id: str) -> List[Detection]:
        """
        Roda a inferência em um único frame (BGR, formato OpenCV) e retorna
        a lista de detecções cuja classe seja de interesse (celular).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class YOLODetector(Detector):
    """
    Detector baseado em Ultralytics YOLO.

    Por padrão usa pesos pré-treinados no COCO (ex.: yolov8n.pt), que já
    reconhecem a classe genérica "cell phone" com precisão limitada em
    ambientes reais de sala de aula. Isso é apenas um ponto de partida:
    NÃO assuma que este modelo genérico terá desempenho adequado em
    produção — ele deve ser validado (ou re-treinado, ver model/training/)
    com dados representativos do ambiente escolar antes de qualquer uso real.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        target_classes: set[str] | None = None,
        min_confidence: float = 0.5,
        inference_resize_width: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "O pacote 'ultralytics' não está instalado. Adicione-o ao requirements.txt "
                "e execute `pip install -r requirements.txt`."
            ) from exc

        self._model = YOLO(model_path)
        self._target_classes = {c.lower() for c in (target_classes or COCO_PHONE_CLASS_NAMES)}
        self._min_confidence = min_confidence
        self._resize_width = inference_resize_width
        self._model_path = model_path

        # Mapa id->nome de classe fornecido pelo próprio modelo carregado.
        self._class_names = {int(k): str(v).lower() for k, v in self._model.names.items()}

    @property
    def name(self) -> str:
        return f"YOLODetector({Path(self._model_path).name})"

    def _resize_if_needed(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        if width <= self._resize_width:
            return frame, 1.0
        import cv2

        scale = self._resize_width / float(width)
        resized = cv2.resize(frame, (self._resize_width, int(height * scale)))
        return resized, scale

    def detect(self, frame: np.ndarray, frame_number: int, camera_id: str) -> List[Detection]:
        resized, scale = self._resize_if_needed(frame)
        results = self._model.predict(resized, verbose=False, conf=self._min_confidence)

        detections: List[Detection] = []
        now = datetime.now(timezone.utc)

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self._class_names.get(cls_id, "unknown")
                if class_name not in self._target_classes:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) / scale for v in box.xyxy[0].tolist())

                detections.append(
                    Detection(
                        detected_class="cell_phone",
                        confidence=confidence,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        timestamp=now,
                        frame_number=frame_number,
                        camera_id=camera_id,
                    )
                )

        return detections


class CustomSchoolPhoneDetector(Detector):
    """
    Placeholder para um modelo treinado especificamente para o domínio
    escolar (ver model/training/). Reutiliza a mesma infraestrutura de
    inferência do YOLODetector, mas aponta para pesos customizados e,
    futuramente, pode incluir pós-processamento específico (ex.: classificar
    "celular em uso" vs "celular sobre a mesa", sem jamais identificar
    pessoas).
    """

    def __init__(self, weights_path: str, min_confidence: float = 0.5, inference_resize_width: int = 640):
        if not Path(weights_path).exists():
            logger.warning(
                "Pesos customizados '%s' não encontrados. Treine um modelo com "
                "model/training/train.py antes de usar este detector.",
                weights_path,
            )
        self._delegate = YOLODetector(
            model_path=weights_path,
            target_classes={"cell_phone", "cell phone"},
            min_confidence=min_confidence,
            inference_resize_width=inference_resize_width,
        )

    @property
    def name(self) -> str:
        return f"CustomSchoolPhoneDetector({self._delegate.name})"

    def detect(self, frame: np.ndarray, frame_number: int, camera_id: str) -> List[Detection]:
        return self._delegate.detect(frame, frame_number, camera_id)


def build_detector(model_path: str, min_confidence: float, inference_resize_width: int) -> Detector:
    """
    Factory simples: se o caminho do modelo parecer um checkpoint customizado
    (convention: contém 'custom' no nome ou vive fora dos pesos padrão do
    Ultralytics), usa CustomSchoolPhoneDetector; caso contrário, YOLODetector
    genérico. Isso pode ser refinado depois sem alterar quem consome
    `Detector`.
    """
    if "custom" in Path(model_path).name.lower():
        return CustomSchoolPhoneDetector(
            weights_path=model_path,
            min_confidence=min_confidence,
            inference_resize_width=inference_resize_width,
        )
    return YOLODetector(
        model_path=model_path,
        min_confidence=min_confidence,
        inference_resize_width=inference_resize_width,
    )
