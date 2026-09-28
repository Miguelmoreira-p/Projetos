from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import cameras, events, health, settings as settings_routes, live_routes
from app.config import get_settings
from app.security.rate_limit import limiter

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("eduphone_guard")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "API do EduPhone Guard: eventos de detecção de uso de celular para "
        "revisão humana. Não realiza reconhecimento facial nem identificação "
        "de estudantes."
    ),
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Nunca retorna stack traces ou detalhes internos ao cliente. O detalhe
    completo vai para o log do servidor.
    """
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno do servidor."},
    )


app.include_router(health.router)
app.include_router(events.router)
app.include_router(cameras.router)
app.include_router(settings_routes.router)
app.include_router(live_routes.router)
