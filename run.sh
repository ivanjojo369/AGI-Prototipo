#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper para controlar el bench GREEN y el servidor HTTP (A2A).
# Uso:
#   ./run.sh bench        # ejecuta el benchmark GREEN
#   ./run.sh start        # inicia el servidor HTTP (GREEN) en 127.0.0.1:8080
#   ./run.sh start 8090   # inicia en 127.0.0.1:8090
#   ./run.sh stop         # detiene el servidor HTTP
#   ./run.sh status       # muestra estado del servidor HTTP

CMD="${1:-}"
shift || true

if [[ -z "$CMD" ]]; then
  echo "Uso: $0 {bench|start|stop|status} [args...]"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_CTRL="run_ctrl.py"

case "$CMD" in
  bench)
    exec "$PYTHON_BIN" "$RUN_CTRL" bench "$@"
    ;;

  start)
    PORT="${1:-8080}"
    shift || true
    exec "$PYTHON_BIN" "$RUN_CTRL" start --port "$PORT" "$@"
    ;;

  stop)
    exec "$PYTHON_BIN" "$RUN_CTRL" stop
    ;;

  status)
    exec "$PYTHON_BIN" "$RUN_CTRL" status
    ;;

  *)
    echo "Comando desconocido: $CMD"
    echo "Uso: $0 {bench|start|stop|status}"
    exit 1
    ;;
esac
