# DEMO (presentación)
## Inicio rápido
.\demo\start_demo.ps1
# Abre: http://127.0.0.1:8009/presentation.html
# Para cerrar: .\demo\stop_demo.ps1

## Guion (90–120 s)
1) Modelo local (OpenChat-3.5-1210 + llama.cpp) y FastAPI (/chat, memoria “lite”).
2) Guardar 1 fact → preguntar sobre “Latencia”.
3) Mostrar TTFT + Total; Streaming ON; ajustar max_new_tokens si es largo.
4) Cierre: /docs y reflexión opcional si hace falta calidad extra.

## Presets recomendados
- max_new_tokens=100, temperature=0.55–0.6, Streaming ON.
- Si verboso: bajar temperature a 0.5 o tokens a 90.
