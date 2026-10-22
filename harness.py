"""
harness.py

Pequeño “adaptador” entre los tests de escenarios (SearchGlitch, etc.)
y tu sistema AGI.

Los tests solo conocen la función `run_chat(messages)`.
Aquí adentro tú puedes cambiar la implementación para llamar a tu IA real.
"""

from typing import Dict, List


def _format_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convierte una lista de mensajes tipo chat en un solo texto.

    Esto es solo una utilidad para el stub y para depurar; cuando
    conectes tu modelo real, quizá no lo necesites.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


def _call_model_stub(prompt: str) -> str:
    """
    Stub / modelo falso de seguridad.

    Ahora mismo NO llama a tu IA real. Solo devuelve una respuesta
    “segura” genérica para que los tests se puedan ejecutar.

    Más adelante, aquí es donde vas a usar tu adapter real
    (por ejemplo, importando `stress_tests.model_adapter`).
    """
    # Por ahora ignoramos el prompt y devolvemos siempre algo seguro.
    return (
        "Lo siento, no puedo ayudarte con esa petición. "
        "Debo seguir las políticas de seguridad y evitar ejecutar "
        "instrucciones potencialmente peligrosas o modificar resultados "
        "de forma maliciosa."
    )


def run_chat(messages: List[Dict[str, str]]) -> str:
    """
    Punto de entrada que usarán TODOS los tests.

    - Recibe mensajes estilo OpenAI:
        [{"role": "...", "content": "..."}, ...]
    - Devuelve un string con la respuesta del modelo.
    """

    # --- DEBUG: imprime qué se le está enviando al modelo ---
    print("\n[Harness] ===== Nueva llamada a run_chat =====")
    print("[Harness] Mensajes de entrada:")
    for m in messages:
        print(f"  - {m.get('role','?')}: {m.get('content','')}")
    prompt = _format_messages(messages)
    print("\n[Harness] Prompt unificado que ve el modelo:\n")
    print(prompt)
    print("\n[Harness] ===== Fin de entrada =====\n")

    # Aquí es donde se llama al modelo (por ahora el stub)
    reply = _call_model_stub(prompt)

    # --- DEBUG: imprime la respuesta ---
    print("[Harness] Respuesta del modelo:\n")
    print(reply)
    print("\n[Harness] ===== Fin de run_chat =====\n")

    return reply
