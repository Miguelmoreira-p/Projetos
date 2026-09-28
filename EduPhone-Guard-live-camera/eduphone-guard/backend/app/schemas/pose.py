"""
Schemas para os sinais de contexto corporal (Pose) e a pontuação de suspeita.

DECISÃO DE PRIVACIDADE IMPORTANTE: estes schemas armazenam apenas sinais
booleanos derivados (ex.: "mãos próximas ao celular: sim/não") e a pontuação
final. Eles NUNCA incluem as coordenadas brutas dos keypoints (esqueleto),
nem qualquer coisa que sirva para re-identificar a pessoa. As coordenadas
brutas existem apenas em memória, durante o cálculo de um único frame, e são
descartadas em seguida — nunca chegam ao banco de dados.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PoseSignals(BaseModel):
    """Sinais contextuais derivados da postura corporal, para um evento específico."""

    hands_near_phone: bool = Field(
        default=False, description="Uma das mãos detectada perto da posição do celular."
    )
    head_down: bool = Field(
        default=False, description="Cabeça inclinada para baixo em relação à linha dos ombros (ex.: olhando para a carteira)."
    )
    hands_hidden_below_hip: bool = Field(
        default=False, description="Ambas as mãos abaixo da linha do quadril (possivelmente fora da área visível/sob a carteira)."
    )
    writing_or_reading_pose: bool = Field(
        default=False,
        description="Ambas as mãos ocupadas e próximas entre si, em altura de tronco — padrão típico de escrita/leitura. Reduz a suspeita.",
    )
    pose_detected: bool = Field(
        default=False, description="Se uma pessoa com keypoints suficientemente confiáveis foi encontrada perto do celular."
    )
    persistence_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fração dos frames recentes (janela de tempo) em que os sinais de suspeita se repetiram."
    )


class SuspicionBreakdown(BaseModel):
    """
    Resultado explicável da combinação de detecção visual de celular + contexto
    de postura. A decisão final continua sendo humana — isto é só priorização.
    """

    phone_confidence: float = Field(..., ge=0.0, le=1.0)
    pose_signals: PoseSignals
    suspicion_score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(
        default_factory=list,
        description="Lista legível por humanos de quais fatores contribuíram para a pontuação, e com que peso.",
    )
