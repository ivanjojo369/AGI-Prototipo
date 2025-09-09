# Troubleshooting
- **WinError 10061**: la API no corre → `uvicorn ... --port 8010`
- **WinError 10013**: puerto ocupado → `netstat -aon | findstr :8010` + `taskkill /PID <pid> /F`
- **/chat falla en CMD**: escapar `|` → `"<^|end_of_turn^|>"`
- **PowerShell JSON**: usa comillas simples o `--data-binary @chat.json`
- **model_not_loaded**: verifica `MODEL_PATH` y `N_GPU_LAYERS`; en CPU usa `0`.
