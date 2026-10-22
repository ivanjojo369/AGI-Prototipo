"""
Quipu Loop - Perfil histórico de compilación de llama-cpp-python
================================================================

Este archivo conserva, en forma compacta y reutilizable, la información útil
extraída del antiguo archivo `build_llama_cpp.log`.

IMPORTANTE:
- No intenta compilar ni instalar nada automáticamente.
- No sustituye al log original como evidencia completa.
- Su función es preservar el estado observado, facilitar diagnósticos futuros
  y evitar conservar un log de casi 2 MB dentro de la raíz del proyecto.

Datos observados en el log:
- Python: 3.12 dentro del entorno Conda `llama-py312`
- pip: 25.3
- Paquete solicitado: llama-cpp-python[server]
- Fuente descargada: llama-cpp-python 0.3.16
- CMake usado durante el build: 4.2.1
- Generador: Visual Studio 17 2022
- Compilador host MSVC: 19.44.35213.0
- CPU backend: AVX2 + FMA + F16C
- AVX-512: no disponible en las pruebas del build
- CUDA Toolkit detectado: 12.4.99
- CUDA backend: detectado e incluido
- Arquitectura CUDA solicitada por CMake: native
- Error determinante observado:
      nvcc fatal: Unsupported gpu architecture 'compute_120'
- Resultado final:
      CMake build failed
      Failed building wheel for llama-cpp-python
      exit code: 1
      9 advertencias / 98 errores

El log NO mostraba CMAKE_ARGS ni FORCE_CMAKE explícitos.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class HistoricalBuildProfile:
    python_environment: str = "llama-py312"
    python_version: str = "3.12"
    pip_version: str = "25.3"

    requested_package: str = "llama-cpp-python[server]"
    observed_source_version: str = "0.3.16"
    install_command: str = (
        'pip install -v --no-cache-dir "llama-cpp-python[server]"'
    )

    cmake_version: str = "4.2.1"
    cmake_generator: str = "Visual Studio 17 2022"
    msvc_host_version: str = "19.44.35213.0"

    cpu_architecture: str = "AMD64 / x86"
    cpu_backend_flags: tuple[str, ...] = ("AVX2", "FMA", "F16C")
    avx512_detected: bool = False

    cuda_toolkit_version: str = "12.4.99"
    cuda_toolkit_root: str = (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    )
    nvcc_path: str = (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
    )
    cuda_architecture_mode: str = "native"
    cuda_architecture_rejected: str = "compute_120"
    cuda_backend_included: bool = True

    ggml_version: str = "0.0.1"
    ggml_commit: str = "4227c9b"

    final_status: str = "FAILED"
    final_exit_code: int = 1
    warning_count: int = 9
    error_count: int = 98
    root_error: str = "nvcc fatal: Unsupported gpu architecture 'compute_120'"


PROFILE = HistoricalBuildProfile()


def _run(command: list[str]) -> dict[str, Optional[str]]:
    """Ejecuta un comando de diagnóstico sin modificar el sistema."""
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": " ".join(command),
            "found": "no",
            "output": None,
            "error": f"No se encontró {command[0]} en PATH",
        }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        return {
            "command": " ".join(command),
            "found": "yes",
            "output": output,
            "error": None if completed.returncode == 0
            else f"exit code {completed.returncode}",
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "found": "yes",
            "output": None,
            "error": repr(exc),
        }


def historical_summary() -> dict:
    """Devuelve los datos históricos conservados del log."""
    return asdict(PROFILE)


def current_environment_snapshot() -> dict:
    """
    Inspecciona el entorno actual sin instalar, borrar ni modificar paquetes.
    """
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "commands": {
            "pip": _run([sys.executable, "-m", "pip", "--version"]),
            "cmake": _run(["cmake", "--version"]),
            "nvcc": _run(["nvcc", "--version"]),
            "nvidia_smi": _run(["nvidia-smi"]),
        },
    }


def explain_failure() -> str:
    """
    Explica únicamente lo que demuestra el log, sin asumir una solución.
    """
    return (
        "El build sí detectó CUDA 12.4.99 e incluyó el backend CUDA. "
        "CMake utilizó la arquitectura CUDA en modo 'native'. "
        "Durante la compilación, nvcc rechazó la arquitectura "
        "'compute_120'. Como consecuencia, CMake terminó con error y "
        "pip no pudo construir el wheel de llama-cpp-python. "
        "Este archivo no presupone qué versión de CUDA o qué configuración "
        "de arquitectura debe usarse para corregirlo; eso debe verificarse "
        "contra el hardware y el entorno actuales."
    )


def print_report(include_current_environment: bool = True) -> None:
    print("=== PERFIL HISTÓRICO DE BUILD ===")
    print(json.dumps(historical_summary(), indent=2, ensure_ascii=False))

    print("\n=== DIAGNÓSTICO CONSERVADO ===")
    print(explain_failure())

    if include_current_environment:
        print("\n=== ENTORNO ACTUAL (SOLO LECTURA) ===")
        print(
            json.dumps(
                current_environment_snapshot(),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    print_report(include_current_environment=True)
