"""
Configuração central da aplicação.

Todos os parâmetros sensíveis (chaves, segredos) e todos os parâmetros de
comportamento (limiares de detecção, retenção, buffers) vivem aqui e são
carregados de variáveis de ambiente / arquivo .env. Nunca deve haver segredos
hardcoded no código-fonte.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "EduPhone Guard"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Supabase ---
    supabase_url: str = Field(default="")
    supabase_anon_key: str = Field(default="")
    supabase_service_role_key: str = Field(default="")
    supabase_jwt_secret: str = Field(default="")
    supabase_media_bucket: str = Field(default="event-clips")

    # --- CORS ---
    # Annotated com NoDecode: sem isso, o pydantic-settings tenta interpretar
    # o valor da variável de ambiente como JSON antes do nosso validador
    # rodar, e quebra porque no .env é uma string simples separada por vírgula.
    cors_allowed_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # --- Visão computacional ---
    model_path: str = Field(default="yolov8n.pt")
    target_classes: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["cell phone"])
    min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_detection_frames: int = Field(default=8, ge=1)
    window_seconds: float = Field(default=3.0, gt=0)
    cooldown_seconds: float = Field(default=30.0, ge=0)
    detection_frame_stride: int = Field(default=1, ge=1)
    inference_resize_width: int = Field(default=480, ge=64)
    live_camera_index: int = Field(default=0, ge=0)
    live_jpeg_quality: int = Field(default=82, ge=40, le=95)
    live_max_fps: float = Field(default=12.0, gt=1, le=30)

    # --- Pose (contexto corporal, complementar ao detector de celular) ---
    # IMPORTANTE: o Pose NUNCA cria um evento sozinho. Ele só roda sobre um
    # frame que já teve uma detecção de celular acima de min_confidence, e
    # serve apenas para ajustar a pontuação de suspeita (reforçar ou reduzir),
    # nunca para gerar suspeita a partir do nada.
    pose_enabled: bool = Field(default=False)
    pose_model_path: str = Field(default="yolov8n-pose.pt")
    pose_min_keypoint_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_hand_proximity_ratio: float = Field(
        default=0.6, gt=0, description="Distância mão-celular considerada 'próxima', como fração da largura dos ombros."
    )
    pose_head_down_ratio: float = Field(
        default=0.12, description="Quanto o nariz precisa estar abaixo da linha dos ombros (fração da altura do tronco) para contar como 'cabeça baixa'."
    )
    pose_writing_hands_together_ratio: float = Field(
        default=0.5, description="Distância máxima entre os punhos (fração da largura dos ombros) para contar como 'mãos ocupadas juntas' (escrevendo/lendo)."
    )

    # Pesos do sistema de pontuação de suspeita (explicável, somável).
    # A pontuação final é limitada a [0, 1] — é um heurístico de priorização
    # para revisão humana, não uma probabilidade calibrada.
    weight_phone_visual: float = Field(default=0.55, ge=0)
    weight_hand_proximity: float = Field(default=0.20, ge=0)
    weight_head_down: float = Field(default=0.10, ge=0)
    weight_hands_hidden: float = Field(default=0.10, ge=0)
    weight_persistence: float = Field(default=0.10, ge=0)
    weight_writing_reduction: float = Field(default=0.25, ge=0)

    # --- Buffer / clipe de vídeo ---
    buffer_seconds: float = Field(default=5.0, gt=0)
    post_event_seconds: float = Field(default=5.0, ge=0)
    clip_output_dir: str = Field(default="./data/clips")
    clip_max_resolution_width: int = Field(default=960, ge=64)

    # --- Retenção / privacidade ---
    retention_days: int = Field(default=7, ge=0)
    retention_sweep_interval_hours: float = Field(default=6.0, gt=0)

    # --- Simulação ---
    simulation_mode: bool = Field(default=True)
    simulation_video_path: str = Field(default="./data/samples/sample.mp4")

    # --- Segurança ---
    rate_limit_default: str = Field(default="60/minute")
    signed_url_ttl_seconds: int = Field(default=300, ge=30)

    @field_validator("cors_allowed_origins", "target_classes", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def clip_output_path(self) -> Path:
        path = Path(self.clip_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
