#!/usr/bin/env python
"""
run_ctrl.py — Pequeño orquestador para AgentBeats (GREEN).

Comandos:
    python run_ctrl.py bench
    python run_ctrl.py start --port 8080
    python run_ctrl.py stop
    python run_ctrl.py status
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
PID_DIR = ROOT / ".agentbeats"
PID_FILE = PID_DIR / "green_server.pid"


def load_env_local() -> Dict[str, str]:
    """
    Carga variables desde .env.local (formato KEY=VALUE, sin BOM, UTF-8).
    No pisa las existentes en os.environ, solo añade faltantes.
    """
    env_path = ROOT / ".env.local"
    result: Dict[str, str] = {}
    if not env_path.exists():
        return result

    text = env_path.read_text(encoding="utf-8").splitlines()
    for line in text:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def run_bench(extra_args: list[str] | None = None) -> None:
    """
    Ejecuta el benchmark GREEN usando tu runner actual.
    Mantiene t1..t18, citas [doc_id], no-answer rule, etc.
    """
    extra_args = extra_args or []
    env = os.environ.copy()
    env.update(load_env_local())

    runner_path = ROOT / "green_agent" / "runner.py"
    if not runner_path.exists():
        raise SystemExit(f"[ERROR] No se encontró {runner_path}")

    cmd = [sys.executable, str(runner_path), *extra_args]
    print(f"[GREEN bench] Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def _server_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return False

    try:
        # Señal 0: solo prueba si el proceso existe (POSIX).
        # En Windows también suele funcionar para verificar existencia.
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def start_server(port: int) -> None:
    """
    Lanza green_agent/server.py como proceso separado.
    Escribe el PID en .agentbeats/green_server.pid
    """
    PID_DIR.mkdir(parents=True, exist_ok=True)

    if _server_running():
        pid = PID_FILE.read_text(encoding="utf-8").strip()
        print(f"[GREEN server] Ya está corriendo (pid={pid}).")
        return

    env = os.environ.copy()
    env.update(load_env_local())
    env.setdefault("AGENTBEATS_HOST", "127.0.0.1")
    env.setdefault("AGENTBEATS_PORT", str(port))

    server_path = ROOT / "green_agent" / "server.py"
    if not server_path.exists():
        raise SystemExit(f"[ERROR] No se encontró {server_path}")

    cmd = [sys.executable, str(server_path)]
    print(f"[GREEN server] Iniciando: {' '.join(cmd)} en {env['AGENTBEATS_HOST']}:{env['AGENTBEATS_PORT']}")

    # Popen: no bloquea, el servidor queda en background.
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=None,
        stderr=None,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"[GREEN server] PID={proc.pid}. Logs en esta misma consola/proceso hijo.")


def stop_server() -> None:
    """
    Detiene el servidor GREEN usando el PID guardado.
    """
    if not PID_FILE.exists():
        print("[GREEN server] No hay PID registrado; ¿ya está detenido?")
        return

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        print("[GREEN server] PID inválido en el archivo, eliminando.")
        PID_FILE.unlink(missing_ok=True)
        return

    print(f"[GREEN server] Deteniendo proceso pid={pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"[GREEN server] Aviso: {exc!r}")
    finally:
        PID_FILE.unlink(missing_ok=True)
        print("[GREEN server] Marcado como detenido.")


def status_server() -> None:
    """
    Muestra el estado del servidor GREEN.
    """
    if not PID_FILE.exists():
        print("[GREEN server] Estado: detenido (sin PID).")
        return

    pid_text = PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_text)
    except ValueError:
        print(f"[GREEN server] Estado: PID inválido en archivo ({pid_text!r}).")
        return

    if _server_running():
        port = os.environ.get("AGENTBEATS_PORT", "8080")
        print(f"[GREEN server] Activo (pid={pid}) en http://127.0.0.1:{port}")
    else:
        print(f"[GREEN server] No responde (pid={pid} parece muerto).")
        print("               Puedes ejecutar: python run_ctrl.py start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentBeats run_ctrl (GREEN)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_bench = subparsers.add_parser("bench", help="Ejecutar benchmark GREEN")
    p_bench.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="Argumentos extra para green_agent/runner.py (opcional).",
    )

    p_start = subparsers.add_parser("start", help="Iniciar servidor HTTP de GREEN")
    p_start.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENTBEATS_PORT", "8080")),
        help="Puerto TCP para el servidor (default: 8080).",
    )

    subparsers.add_parser("stop", help="Detener servidor HTTP de GREEN")
    subparsers.add_parser("status", help="Estado del servidor HTTP de GREEN")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "bench":
        run_bench(args.bench_args)

    elif args.command == "start":
        start_server(args.port)

    elif args.command == "stop":
        stop_server()

    elif args.command == "status":
        status_server()

    else:
        raise SystemExit(f"Comando desconocido: {args.command}")


if __name__ == "__main__":
    main()
