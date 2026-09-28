"""Monitoramento local em tempo real para o protótipo do EduPhone Guard."""
from __future__ import annotations

import threading
import time
from typing import Generator

import cv2
import numpy as np

from app.config import get_settings
from app.vision.detector import build_detector
from app.vision.event_detector import EventDetector
from app.vision.pose import PoseAnalyzer, SuspicionScorer
from app.video.camera import VideoSource


class LiveMonitor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._camera: VideoSource | None = None
        self._latest_jpeg: bytes | None = None
        self._running = False
        self._error: str | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._phone_count = 0
        self._last_confidence = 0.0
        self._last_event = False
        self._last_suspicion = None
        self._started_at = None

        self._detector = build_detector(
            model_path=self.settings.model_path,
            min_confidence=self.settings.min_confidence,
            inference_resize_width=self.settings.inference_resize_width,
        )

        pose = None
        if self.settings.pose_enabled:
            pose = PoseAnalyzer(
                model_path=self.settings.pose_model_path,
                min_keypoint_confidence=self.settings.pose_min_keypoint_confidence,
                hand_proximity_ratio=self.settings.pose_hand_proximity_ratio,
                head_down_ratio=self.settings.pose_head_down_ratio,
                writing_hands_together_ratio=self.settings.pose_writing_hands_together_ratio,
            )

        scorer = SuspicionScorer(
            weight_phone_visual=self.settings.weight_phone_visual,
            weight_hand_proximity=self.settings.weight_hand_proximity,
            weight_head_down=self.settings.weight_head_down,
            weight_hands_hidden=self.settings.weight_hands_hidden,
            weight_persistence=self.settings.weight_persistence,
            weight_writing_reduction=self.settings.weight_writing_reduction,
        )
        self._events = EventDetector(
            min_confidence=self.settings.min_confidence,
            min_detection_frames=self.settings.min_detection_frames,
            window_seconds=self.settings.window_seconds,
            cooldown_seconds=self.settings.cooldown_seconds,
            pose_analyzer=pose,
            suspicion_scorer=scorer if pose else None,
        )

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop.clear()
            self._error = None
            self._latest_jpeg = None
            self._frame_count = 0
            self._phone_count = 0
            self._last_confidence = 0.0
            self._last_event = False
            self._last_suspicion = None
            self._running = True
            self._started_at = time.time()

        self._thread = threading.Thread(target=self._run, name="eduphone-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._running = False
        self._thread = None

    def _set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._latest_jpeg = jpeg

    def _run(self) -> None:
        started = time.monotonic()
        frames_since = 0
        camera = None
        try:
            camera = VideoSource(self.settings.live_camera_index)
            self._camera = camera
            min_interval = 1.0 / self.settings.live_max_fps
            last_processed = 0.0

            for frame, _, frame_number in camera.frames():
                if self._stop.is_set():
                    break
                now_mono = time.monotonic()
                if now_mono - last_processed < min_interval:
                    continue
                last_processed = now_mono

                detections = self._detector.detect(frame, frame_number, camera_id="live")
                trigger = self._events.process(detections, camera_id="live", frame=frame)

                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox.as_tuple())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 80), 2)
                    label = f"CELULAR {det.confidence:.0%}"
                    cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (x1 + 150, y1), (0, 220, 80), -1)
                    cv2.putText(annotated, label, (x1 + 5, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                status = "MONITORAMENTO ATIVO"
                cv2.rectangle(annotated, (12, 12), (340, 62), (25, 25, 25), -1)
                cv2.putText(annotated, status, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
                cv2.putText(annotated, f"Celulares: {len(detections)}", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if trigger:
                    cv2.putText(annotated, "EVENTO CONFIRMADO - REVISAR", (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.live_jpeg_quality]
                )
                if ok:
                    self._set_frame(encoded.tobytes())

                frames_since += 1
                self._frame_count += 1
                elapsed = time.monotonic() - started
                fps = frames_since / elapsed if elapsed > 0 else 0.0
                with self._lock:
                    self._fps = fps
                    self._phone_count = len(detections)
                    self._last_confidence = max((d.confidence for d in detections), default=0.0)
                    self._last_event = trigger is not None
                    self._last_suspicion = trigger.suspicion.model_dump() if trigger and trigger.suspicion else None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            if camera is not None:
                camera.release()
            self._camera = None
            with self._lock:
                self._running = False

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "frame_count": self._frame_count,
                "fps": round(self._fps, 1),
                "phone_count": self._phone_count,
                "last_confidence": round(self._last_confidence, 3),
                "last_event": self._last_event,
                "last_suspicion": self._last_suspicion,
                "detector": self._detector.name,
                "pose_enabled": self.settings.pose_enabled,
                "camera_index": self.settings.live_camera_index,
            }

    def stream(self) -> Generator[bytes, None, None]:
        while True:
            with self._lock:
                running = self._running
                frame = self._latest_jpeg
            if not running and frame is None:
                break
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)


monitor = LiveMonitor()
