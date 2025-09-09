# SETUP
## Requisitos
- Python 3.10+ · `pip install -r requirements.txt`
- Modelo: OpenChat-3.5-1210 (.gguf) → `MODEL_PATH` en `.env`

## Variables (ejemplo)
- MODEL_PATH=C:\Models\openchat-3.5-1210.Q4_K_M.gguf
- HOST=127.0.0.1
- PORT=8010
- N_CTX=4096
- N_GPU_LAYERS=0  # usa 0 si no tienes build CUDA

## Ejecutar
uvicorn llama_server:app --host 127.0.0.1 --port 8010 --reload
python -m http.server 8009 -d demo

## Pruebas rápidas
curl http://127.0.0.1:8010/
