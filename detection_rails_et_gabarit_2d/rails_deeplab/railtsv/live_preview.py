"""Local browser preview of ongoing processing, without claiming real-time speed."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import cv2


class LivePreview:
    def __init__(self):
        self.jpeg=None;self.info={};self.lock=threading.Lock();owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path=='/':
                    body='''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Traitement TSV en cours</title><style>body{background:#17232e;color:white;font:18px system-ui;margin:25px;line-height:1.5}img{width:100%;max-width:1400px}progress{width:100%;max-width:1400px;height:24px}</style><h1>Traitement TSV en cours</h1><p>L'image s'actualise pendant le calcul. Tu pourras lire la vidéo terminée à sa vitesse normale.</p><progress id="progress" max="100" value="0"></progress><p id="info">Préparation de la première image…</p><p id="speed"></p><img id="frame" alt="Dernière image traitée"><script>setInterval(async()=>{try{let s=await(await fetch('/status')).json();if(!s.images)return;document.getElementById('progress').value=100*s.images/s.total;document.getElementById('info').textContent=`${s.images} / ${s.total} images traitées — ${s.video_seconds.toFixed(1)} secondes de vidéo`;document.getElementById('speed').textContent=`Vitesse du calcul : ${s.processing_fps} images/s. Vitesse de la vidéo : ${s.source_fps} images/s.`;document.getElementById('frame').src='/frame.jpg?t='+Date.now()}catch(e){document.getElementById('speed').textContent='Aperçu arrêté. Le résultat terminé se trouve dans 03_RESULTATS.'}},1000)</script></html>'''.encode('utf-8')
                    mime='text/html; charset=utf-8'
                elif self.path=='/status':
                    with owner.lock:body=json.dumps(owner.info).encode()
                    mime='application/json'
                elif self.path.startswith('/frame.jpg'):
                    with owner.lock:body=owner.jpeg
                    if body is None:self.send_error(503);return
                    mime='image/jpeg'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/'
    def update(self,image,**info):
        h,w=image.shape[:2];preview=cv2.resize(image,(min(1280,w),round(h*min(1280,w)/w)))
        ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(preview,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('Aperçu non encodé')
        with self.lock:self.jpeg=jpeg.tobytes();self.info=info
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
