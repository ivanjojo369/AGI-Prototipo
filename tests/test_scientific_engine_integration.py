from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from quipu_loop import QuipuLoop
from scientific_engine import ScientificEngine


class CountingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return "RESPUESTA_QWEN"


class MemoryRecorder:
    def __init__(self) -> None:
        self.rows = []

    def add(self, role, content, metadata=None):
        self.rows.append(
            (role, content, dict(metadata or {}))
        )


def make_loop(*, use_verifier: bool = True):
    loop = object.__new__(QuipuLoop)
    loop.cfg = SimpleNamespace(
        use_verifier=use_verifier,
        max_tokens=512,
        temperature=0.7,
    )
    loop.adapter = CountingAdapter()
    loop.memory = MemoryRecorder()
    loop.scientific_engine = ScientificEngine()
    loop.factual_verifier = object() if use_verifier else None
    loop.self_correction = None
    loop.conversation_state_manager = None
    loop.output_guard = None
    loop.session_path = (
        Path(tempfile.mkdtemp())
        / "session_test.jsonl"
    )
    loop._last_orchestration = {}
    loop._last_cognitive_state = None
    loop._last_factual_report = {}
    loop._last_scientific_result = {}
    loop._factual_debug_enabled = False
    loop._factual_debug_phase = None
    loop._factual_debug_path = (
        loop.session_path.with_name(
            "session_test_factual_debug.jsonl"
        )
    )

    loop._prepare_cognitive_state = lambda _text: None
    loop._prepare_orchestration = lambda _text: {
        "ok": True,
        "skill": "chat",
        "confidence": 0.50,
        "reason": "test",
        "output": None,
    }
    loop._build_messages = (
        lambda user_text, orchestration=None, cognitive_state=None: [
            {"role": "user", "content": user_text}
        ]
    )
    loop._metacognitive_block_answer = lambda _state: None
    loop._deterministic_quantitative_answer = (
        lambda _text, _state: None
    )
    loop._strip_internal_orchestration_leaks = lambda answer: str(answer)
    loop._apply_output_guard = (
        lambda _state, answer, messages=None, allow_regeneration=False:
            str(answer)
    )
    loop._log_cognitive_state = lambda _state: None

    return loop


def forbid_probabilistic_verifiers(loop):
    loop._verify_answer = lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            AssertionError("Classic Verifier no debe tocar ciencia determinista")
        )
    )
    loop._apply_factual_verification = lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            AssertionError("FactualVerifier/Qwen no debe rejuzgar ciencia determinista")
        )
    )


def test_1_engine_is_available_in_loop():
    loop = make_loop()
    assert loop.scientific_engine is not None
    assert loop.scientific_engine.status()["version"] == "1.1.0"


def test_2_schwarzschild_bypasses_qwen():
    loop = make_loop(use_verifier=True)
    forbid_probabilistic_verifiers(loop)

    answer = loop.run_once(
        "¿Qué es un horizonte de sucesos y dónde se encuentra "
        "respecto a la singularidad en un agujero negro de Schwarzschild?"
    )

    assert loop.adapter.calls == 0
    assert "r_s = 2GM/c^2" in answer
    assert "r = 0" in answer


def test_3_normal_chat_still_uses_qwen():
    loop = make_loop(use_verifier=False)
    answer = loop.run_once("Hola, ¿cómo estás?")

    assert answer == "RESPUESTA_QWEN"
    assert loop.adapter.calls == 1


def test_4_black_hole_radius_uses_deterministic_tool():
    loop = make_loop(use_verifier=True)
    forbid_probabilistic_verifiers(loop)

    answer = loop.run_once(
        "Calcula el radio de Schwarzschild de un agujero negro "
        "de 10 masas solares."
    )

    assert loop.adapter.calls == 0
    assert "29." in answer
    assert "km" in answer


def test_5_output_guard_still_runs_without_regeneration():
    loop = make_loop(use_verifier=True)
    seen = {"called": False, "allow_regeneration": None}

    def guard(_state, answer, messages=None, allow_regeneration=False):
        seen["called"] = True
        seen["allow_regeneration"] = allow_regeneration
        return str(answer)

    loop._apply_output_guard = guard
    loop.run_once("Deriva x³ + 2x")

    assert seen["called"] is True
    assert seen["allow_regeneration"] is False
    assert loop.adapter.calls == 0


def test_6_memory_marks_scientific_authority():
    loop = make_loop(use_verifier=True)
    loop.run_once("Resuelve x² - 4 = 0")

    assistant = [row for row in loop.memory.rows if row[0] == "assistant"][-1]
    metadata = assistant[2]

    assert metadata["scientific_engine"] is True
    assert metadata["deterministic"] is True
    assert metadata["scientific_domain"] == "mathematics"
    assert metadata["scientific_intent"] == "solve_equation"
    assert metadata["scientific_engine_version"] == "1.1.0"


def test_7_old_factual_report_cannot_contaminate_science_turn():
    loop = make_loop(use_verifier=True)
    loop._last_factual_report = {
        "correction_required": True,
        "judge_error_count": 1,
    }

    loop.run_once("¿Qué es el horizonte de sucesos en Schwarzschild?")

    assistant = [row for row in loop.memory.rows if row[0] == "assistant"][-1]
    metadata = assistant[2]

    assert metadata["context_reuse"] is True
    assert metadata.get("cognitive_status") != "factual_verification_blocked"


def test_8_scientific_result_is_logged():
    loop = make_loop(use_verifier=False)
    loop.run_once("Convierte 5 kilómetros a metros")

    content = loop.session_path.read_text(encoding="utf-8")
    assert '"type": "scientific_engine"' in content
    assert '"deterministic_scientific": true' in content
    assert '"orchestrated_by": "scientific_engine"' in content


def test_9_symbolic_math_bypasses_qwen_and_verifiers():
    loop = make_loop(use_verifier=True)
    forbid_probabilistic_verifiers(loop)

    answer = loop.run_once("Resuelve x² - 4 = 0")

    assert loop.adapter.calls == 0
    assert "x = -2" in answer
    assert "x = 2" in answer


def test_10_units_bypass_qwen_and_verifiers():
    loop = make_loop(use_verifier=True)
    forbid_probabilistic_verifiers(loop)

    answer = loop.run_once("Convierte 72 km/h a m/s")

    assert loop.adapter.calls == 0
    assert "20" in answer
    assert "meter/second" in answer


def test_11_metacognitive_block_cannot_preempt_scientific_tool():
    loop = make_loop(use_verifier=False)
    loop._metacognitive_block_answer = lambda _state: "BLOQUEADO_POR_GATE"

    answer = loop.run_once("Deriva x³ + 2x")

    assert answer != "BLOQUEADO_POR_GATE"
    assert "3*x**2 + 2" in answer
    assert loop.adapter.calls == 0


def test_12_non_scientific_turn_still_respects_metacognitive_block():
    loop = make_loop(use_verifier=False)
    loop._metacognitive_block_answer = lambda _state: "BLOQUEADO_POR_GATE"

    answer = loop.run_once("Explícame un tema cualquiera")

    assert answer == "BLOQUEADO_POR_GATE"
    assert loop.adapter.calls == 0


def main():
    tests = [
        test_1_engine_is_available_in_loop,
        test_2_schwarzschild_bypasses_qwen,
        test_3_normal_chat_still_uses_qwen,
        test_4_black_hole_radius_uses_deterministic_tool,
        test_5_output_guard_still_runs_without_regeneration,
        test_6_memory_marks_scientific_authority,
        test_7_old_factual_report_cannot_contaminate_science_turn,
        test_8_scientific_result_is_logged,
        test_9_symbolic_math_bypasses_qwen_and_verifiers,
        test_10_units_bypass_qwen_and_verifiers,
        test_11_metacognitive_block_cannot_preempt_scientific_tool,
        test_12_non_scientific_turn_still_respects_metacognitive_block,
    ]

    for index, test in enumerate(tests, 1):
        test()
        print(f"TEST {index:02d} — OK — {test.__name__}")

    print("=" * 100)
    print("QUIPU SCIENTIFIC ENGINE INTEGRATION 1.1.0: 12/12 pruebas superadas.")


if __name__ == "__main__":
    main()
