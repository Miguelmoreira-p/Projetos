from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from app.api.live import monitor

router = APIRouter(tags=["live-monitor"])


@router.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
def monitor_page() -> str:
    return """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EduPhone Guard — Monitoramento</title>
<style>
body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.bar{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}button{border:0;border-radius:8px;padding:12px 18px;font-weight:700;cursor:pointer}#start{background:#19c37d;color:#08130e}#stop{background:#e05252;color:white}button:disabled{opacity:.45;cursor:not-allowed}.status{padding:8px 12px;border-radius:20px;background:#222}.video{margin-top:18px;background:#000;border-radius:12px;overflow:hidden;min-height:420px;display:flex;align-items:center;justify-content:center}.video img{width:100%;height:auto;display:block}.empty{color:#888;font-size:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.card{background:#1c1c1c;border-radius:10px;padding:14px}.value{font-size:24px;font-weight:700;margin-top:5px}.hint{color:#aaa;margin-top:14px;font-size:14px}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap"><div class="bar"><div><h1>EduPhone Guard</h1><div class="status" id="status">Parado</div></div><div><button id="start">Iniciar monitoramento</button> <button id="stop" disabled>Parar</button></div></div>
<div class="video"><div class="empty" id="empty">Clique em “Iniciar monitoramento” para abrir a câmera.</div><img id="feed" style="display:none" alt="Câmera ao vivo"></div>
<div class="stats"><div class="card">FPS<div class="value" id="fps">0</div></div><div class="card">Celulares no frame<div class="value" id="phones">0</div></div><div class="card">Confiança<div class="value" id="conf">0%</div></div><div class="card">Eventos<div class="value" id="event">Nenhum</div></div></div>
<div class="hint">Modo local de teste. A imagem é processada pelo backend; não há reconhecimento facial ou identificação de alunos.</div></div>
<script>
const start=document.getElementById('start'),stop=document.getElementById('stop'),feed=document.getElementById('feed'),empty=document.getElementById('empty');
async function refresh(){try{const r=await fetch('/live/status');const s=await r.json();document.getElementById('fps').textContent=s.fps;document.getElementById('phones').textContent=s.phone_count;document.getElementById('conf').textContent=Math.round(s.last_confidence*100)+'%';document.getElementById('event').textContent=s.last_event?'CONFIRMADO':'Nenhum';document.getElementById('status').textContent=s.running?'Monitoramento ativo':'Parado';start.disabled=s.running;stop.disabled=!s.running;if(s.error) document.getElementById('status').textContent='Erro: '+s.error;}catch(e){}}
start.onclick=async()=>{await fetch('/live/start',{method:'POST'});feed.src='/live/stream?'+Date.now();feed.style.display='block';empty.style.display='none';refresh();};
stop.onclick=async()=>{await fetch('/live/stop',{method:'POST'});feed.removeAttribute('src');feed.style.display='none';empty.style.display='block';refresh();};
setInterval(refresh,1000);refresh();
</script></body></html>"""


@router.post("/live/start")
def start_live():
    monitor.start()
    return monitor.status()


@router.post("/live/stop")
def stop_live():
    monitor.stop()
    return monitor.status()


@router.get("/live/status")
def live_status():
    return monitor.status()


@router.get("/live/stream")
def live_stream():
    return StreamingResponse(monitor.stream(), media_type="multipart/x-mixed-replace; boundary=frame")
