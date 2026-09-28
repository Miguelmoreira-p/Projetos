"""
Modo de simulação: processa um arquivo de vídeo local ponta a ponta pelo
pipeline completo (detecção -> persistência/cooldown -> clipe -> evento),
sem depender de câmeras reais.

Uso:
    python -m app.simulate --video data/samples/sample.mp4 --camera-id <uuid> --school-id <uuid>

Se --camera-id / --school-id não forem fornecidos, usa UUIDs de exemplo
fixos (apenas para fins de teste local; substitua por IDs reais cadastrados
no Supabase para persistir eventos de verdade).
"""
from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.schemas.event import EventCreate
from app.video.buffer import CircularFrameBuffer
from app.video.camera import VideoSource
from app.video.recorder import ClipRecorder
from app.vision.detector import build_detector
from app.vision.event_detector import EventDetector
from app.vision.pose import PoseAnalyzer, SuspicionScorer

logging.basicConfig(level="INFO")
logger = logging.getLogger("eduphone_guard.simulate")

_DEFAULT_SCHOOL_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_DEFAULT_CAMERA_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def run_simulation(video_path: str, school_id: uuid.UUID, camera_id: uuid.UUID, persist: bool) -> None:
    settings = get_settings()

    detector = build_detector(
        model_path=settings.model_path,
        min_confidence=settings.min_confidence,
        inference_resize_width=settings.inference_resize_width,
    )
    logger.info("using_detector name=%s", detector.name)

    pose_analyzer = None
    suspicion_scorer = None
    if settings.pose_enabled:
        pose_analyzer = PoseAnalyzer(
            model_path=settings.pose_model_path,
            min_keypoint_confidence=settings.pose_min_keypoint_confidence,
            hand_proximity_ratio=settings.pose_hand_proximity_ratio,
            head_down_ratio=settings.pose_head_down_ratio,
            writing_hands_together_ratio=settings.pose_writing_hands_together_ratio,
        )
        suspicion_scorer = SuspicionScorer(
            weight_phone_visual=settings.weight_phone_visual,
            weight_hand_proximity=settings.weight_hand_proximity,
            weight_head_down=settings.weight_head_down,
            weight_hands_hidden=settings.weight_hands_hidden,
            weight_persistence=settings.weight_persistence,
            weight_writing_reduction=settings.weight_writing_reduction,
        )
        if not pose_analyzer.available:
            logger.warning("POSE_ENABLED=true mas o modelo de pose não carregou; seguindo sem contexto de pose.")

    event_detector = EventDetector(
        min_confidence=settings.min_confidence,
        min_detection_frames=settings.min_detection_frames,
        window_seconds=settings.window_seconds,
        cooldown_seconds=settings.cooldown_seconds,
        pose_analyzer=pose_analyzer,
        suspicion_scorer=suspicion_scorer,
    )
    recorder = ClipRecorder(output_dir=str(settings.clip_output_path), max_width=settings.clip_max_resolution_width)

    camera_id_str = str(camera_id)
    events_created = 0

    with VideoSource(video_path) as source:
        frame_buffer = CircularFrameBuffer(buffer_seconds=settings.buffer_seconds, fps_estimate=source.fps)
        post_event_frames_remaining = 0
        post_event_frames: list = []
        pending_trigger = None

        for frame, elapsed, frame_number in source.frames():
            if frame_number % settings.detection_frame_stride != 0:
                frame_buffer.append(frame, elapsed, frame_number)
                continue

            detections = detector.detect(frame, frame_number, camera_id_str)
            frame_buffer.append(frame, elapsed, frame_number)

            if post_event_frames_remaining > 0:
                post_event_frames.append(frame)
                post_event_frames_remaining -= 1
                if post_event_frames_remaining == 0 and pending_trigger is not None:
                    _finalize_event(
                        recorder=recorder,
                        frame_buffer=frame_buffer,
                        post_event_frames=post_event_frames,
                        school_id=school_id,
                        camera_id=camera_id,
                        trigger=pending_trigger,
                        persist=persist,
                    )
                    events_created += 1
                    post_event_frames = []
                    pending_trigger = None
                continue

            trigger = event_detector.process(detections, camera_id_str, frame=frame, now=datetime.now(timezone.utc))
            if trigger is not None:
                logger.info("trigger_detected frame=%d confidence=%.3f", frame_number, trigger.confidence)
                pending_trigger = trigger
                post_event_frames_remaining = max(1, int(settings.post_event_seconds * source.fps))

        # Se o vídeo terminou com um evento pendente de pós-captura, finaliza com o que temos.
        if pending_trigger is not None:
            _finalize_event(
                recorder=recorder,
                frame_buffer=frame_buffer,
                post_event_frames=post_event_frames,
                school_id=school_id,
                camera_id=camera_id,
                trigger=pending_trigger,
                persist=persist,
            )
            events_created += 1

    logger.info("simulation_finished events_created=%d", events_created)


def _build_detection_metadata(trigger) -> dict:
    """
    Monta o detection_metadata do evento. Sempre inclui phone_confidence.
    Quando o Pose está habilitado e produziu um resultado, inclui também
    pose_signals, suspicion_score e reasons — separadamente, como pedido,
    nunca misturados de forma que pareçam a mesma coisa que a detecção
    visual do celular.
    """
    metadata: dict = {"source": "simulation", "phone_confidence": round(trigger.confidence, 4)}

    if trigger.suspicion is not None:
        metadata["pose_signals"] = trigger.suspicion.pose_signals.model_dump()
        metadata["suspicion_score"] = round(trigger.suspicion.suspicion_score, 4)
        metadata["suspicion_reasons"] = trigger.suspicion.reasons

    return metadata


def _finalize_event(recorder, frame_buffer, post_event_frames, school_id, camera_id, trigger, persist: bool) -> None:
    event_id = uuid.uuid4()
    clip_path = recorder.write_clip(
        event_id=event_id,
        pre_frames=frame_buffer.snapshot(),
        post_frames=post_event_frames,
    )
    logger.info("clip_ready event_id=%s path=%s", event_id, clip_path)

    if not persist:
        logger.info("persist=False: evento não enviado ao Supabase (apenas clipe local gerado).")
        return

    from app.services import event_service, storage_service

    storage_path = f"{school_id}/{event_id}.mp4"
    try:
        storage_service.upload_clip(clip_path, storage_path)
    except Exception:
        logger.exception("failed_to_upload_clip event_id=%s", event_id)
        storage_path = None

    event_service.create_event(
        EventCreate(
            school_id=school_id,
            camera_id=camera_id,
            detected_at=datetime.now(timezone.utc),
            confidence=trigger.confidence,
            video_path=storage_path,
            detection_metadata=_build_detection_metadata(trigger),
        )
    )
    recorder.delete_clip(clip_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula o pipeline completo do EduPhone Guard sobre um arquivo de vídeo.")
    parser.add_argument("--video", required=True, help="Caminho para o arquivo MP4 de entrada.")
    parser.add_argument("--school-id", default=str(_DEFAULT_SCHOOL_ID))
    parser.add_argument("--camera-id", default=str(_DEFAULT_CAMERA_ID))
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Não envia eventos ao Supabase; apenas gera clipes locais para inspeção.",
    )
    args = parser.parse_args()

    run_simulation(
        video_path=args.video,
        school_id=uuid.UUID(args.school_id),
        camera_id=uuid.UUID(args.camera_id),
        persist=not args.no_persist,
    )


if __name__ == "__main__":
    main()
