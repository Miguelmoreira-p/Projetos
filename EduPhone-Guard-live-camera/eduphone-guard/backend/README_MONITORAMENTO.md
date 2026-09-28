# Monitoramento ao vivo

Foi adicionado um modo local de teste em `/monitor`.

1. No terminal, entre na pasta `backend`.
2. Instale as dependências se ainda não instalou: `pip install -r requirements.txt`.
3. Rode `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` ou abra `INICIAR_MONITORAMENTO.bat`.
4. Abra `http://127.0.0.1:8000/monitor` no navegador.
5. Clique em **Iniciar monitoramento**.
6. A câmera `LIVE_CAMERA_INDEX` (padrão `0`) será aberta pelo OpenCV e o vídeo anotado será transmitido para a página.

Configurações novas no `.env`:

```env
LIVE_CAMERA_INDEX=0
LIVE_JPEG_QUALITY=82
LIVE_MAX_FPS=30
```

O monitoramento usa o mesmo `MODEL_PATH`, `MIN_CONFIDENCE`, `MIN_DETECTION_FRAMES` e `INFERENCE_RESIZE_WIDTH` já existentes. O Pose continua opcional via `POSE_ENABLED=true`.

O modo é local para prototipagem: a câmera é processada no computador que executa o backend. Não há reconhecimento facial nem identificação de estudantes.
