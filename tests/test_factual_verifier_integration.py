from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

from quipu_loop import LoopConfig, QuipuLoop
from verifier.factual_verifier import FactualVerifier


EVIDENCE = [{
    "id": "src_schwarzschild_1",
    "text": (
        "Para un agujero negro de Schwarzschild, el horizonte de sucesos "
        "está en r = 2GM/c^2. La singularidad clásica está en r = 0; "
        "no son la misma ubicación."
    ),
    "score": 0.95,
    "meta": {
        "path": "docs/black_holes.md",
        "authority": 0.95,
        "chunk": 2,
    },
}]


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.normal_drafts = 0

    def generate(
        self,
        messages,
        max_tokens=512,
        temperature=0.7,
        stream=False,
        stop=None,
    ):
        self.calls.append(list(messages))
        rendered = "\n".join(
            str(item.get("content") or "")
            for item in messages
        )

        if "juez factual basado estrictamente en evidencia" in rendered:
            payload = json.loads(messages[-1]["content"])
            claim = payload["claim"].lower()
            evidence = " ".join(
                str(item.get("text") or "").lower()
                for item in payload.get("evidence", [])
            )

            if (
                "horizonte" in claim
                and "singularidad" in claim
                and "coincide" in claim
                and "no son la misma" in evidence
            ):
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.99,
                    "reason": "La evidencia distingue horizonte y singularidad.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            if (
                "horizonte" in claim
                and "2gm/c^2" in claim.replace(" ", "")
                and "r = 2gm/c^2" in evidence
            ):
                return json.dumps({
                    "status": "supported",
                    "confidence": 0.98,
                    "reason": "Respaldado por la evidencia.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            if (
                "singularidad" in claim
                and "r = 0" in claim
                and "singularidad clásica está en r = 0" in evidence
            ):
                return json.dumps({
                    "status": "supported",
                    "confidence": 0.98,
                    "reason": "Respaldado por la evidencia.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            return json.dumps({
                "status": "uncertain",
                "confidence": 0.50,
                "reason": "La evidencia no basta.",
                "evidence_ids": ["src_schwarzschild_1"],
            })

        if "Corrección factual obligatoria" in rendered:
            return (
                "En Schwarzschild, el horizonte de sucesos está en "
                "r = 2GM/c^2. La singularidad clásica está en r = 0."
            )

        self.normal_drafts += 1
        return "El horizonte de sucesos coincide con la singularidad."


class AlwaysWrongAdapter(FakeAdapter):
    def generate(self, messages, **kwargs):
        rendered = "\n".join(
            str(item.get("content") or "")
            for item in messages
        )
        if "juez factual basado estrictamente en evidencia" in rendered:
            return super().generate(messages, **kwargs)
        self.calls.append(list(messages))
        return "El horizonte de sucesos coincide con la singularidad."


class RealFailureReproductionAdapter(FakeAdapter):
    """
    Reproduce el fallo observado el 2026-10-05:
    la segunda pasada marca como CONTRADICTED una frase ambigua,
    pero no cita evidence_ids.
    """

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        rendered = "\n".join(
            str(item.get("content") or "")
            for item in messages
        )

        if "juez factual basado estrictamente en evidencia" in rendered:
            payload = json.loads(messages[-1]["content"])
            claim = payload["claim"].lower()

            if "coincide con la singularidad" in claim:
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.99,
                    "reason": "Horizonte y singularidad son regiones distintas.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            if "no está relacionado directamente" in claim:
                # Exactamente el patrón problemático del log real:
                # contradicted fuerte SIN atribución de evidence_ids.
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.98,
                    "reason": (
                        "Las regiones son distintas, por lo que considero "
                        "contradicha la relación."
                    ),
                })

            if (
                "r = 0" in claim
                or "2gm/c^2" in claim.replace(" ", "")
                or "barrera infranqueable" in claim
            ):
                return json.dumps({
                    "status": "supported",
                    "confidence": 0.98,
                    "reason": "Respaldado por la evidencia.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            return json.dumps({
                "status": "uncertain",
                "confidence": 0.50,
                "reason": "No basta la evidencia.",
                "evidence_ids": ["src_schwarzschild_1"],
            })

        if "Corrección factual obligatoria" in rendered:
            return (
                "El horizonte de sucesos en un agujero negro de Schwarzschild "
                "es una barrera infranqueable desde nuestro punto de vista en "
                "el exterior del agujero negro. Se encuentra en la región donde "
                "el radio de Schwarzschild (r_s = 2GM/c^2) se alcanza, y no está "
                "relacionado directamente con la singularidad en el centro del "
                "agujero negro. La singularidad está en el centro, a r = 0, "
                "mientras que el horizonte de sucesos ocurre en r = r_s."
            )

        return "El horizonte de sucesos coincide con la singularidad."


class ValidIdFalseContradictionAdapter(FakeAdapter):
    """
    Reproduce el fallo real:
    Qwen devuelve CONTRADICTED + evidence_id válido cuando la evidencia
    en realidad apoya la separación horizonte/singularidad.
    """

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        rendered = "\n".join(
            str(item.get("content") or "")
            for item in messages
        )

        if "juez factual basado estrictamente en evidencia" in rendered:
            payload = json.loads(messages[-1]["content"])
            claim = payload["claim"].lower()

            if "coincide con la singularidad" in claim:
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.99,
                    "reason": "La evidencia dice que no son la misma ubicación.",
                    "evidence_ids": ["src_schwarzschild_1"],
                    "evidence_quote": "no son la misma ubicación.",
                    "claim_fact": "El horizonte coincide con la singularidad.",
                    "evidence_fact": "no son la misma ubicación.",
                    "contradiction_type": "identity_conflict",
                })

            if (
                "regiones distintas" in claim
                or "region separada" in claim
                or "región separada" in claim
                or "no llega hasta la singularidad" in claim
            ):
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.98,
                    "reason": "Interpreto erróneamente separación como conflicto.",
                    "evidence_ids": ["src_schwarzschild_1"],
                    "evidence_quote": "no son la misma ubicación.",
                    "claim_fact": "El horizonte y la singularidad son regiones distintas.",
                    "evidence_fact": "no son la misma ubicación.",
                    "contradiction_type": "identity_conflict",
                })

            if (
                "r = 0" in claim
                or "2gm/c^2" in claim.replace(" ", "")
                or "barrera infranqueable" in claim
            ):
                return json.dumps({
                    "status": "supported",
                    "confidence": 0.98,
                    "reason": "Respaldado por la evidencia.",
                    "evidence_ids": ["src_schwarzschild_1"],
                })

            return json.dumps({
                "status": "uncertain",
                "confidence": 0.50,
                "reason": "La evidencia no basta.",
                "evidence_ids": ["src_schwarzschild_1"],
            })

        if "Corrección factual obligatoria" in rendered:
            return (
                "En Schwarzschild, el horizonte de sucesos está en "
                "r = 2GM/c^2. La singularidad clásica está en r = 0. "
                "El horizonte y la singularidad son regiones distintas."
            )

        return "El horizonte de sucesos coincide con la singularidad."


class FakeClassicVerifier:
    def verify(self, answer, messages, evidence=None, structured_facts=None):
        return str(answer)


class FakeMemory:
    def __init__(self):
        self.saved = []

    def add(self, role, content, metadata=None):
        self.saved.append((role, content, dict(metadata or {})))


class FakeCfg:
    use_rag = True
    use_verifier = True
    max_tokens = 512
    temperature = 0.7


def evidence_provider(_claim, top_k):
    return EVIDENCE[:top_k]


def make_loop(adapter=None):
    loop = object.__new__(QuipuLoop)
    loop.cfg = FakeCfg()
    loop.adapter = adapter or FakeAdapter()
    loop.verifier = FakeClassicVerifier()
    loop.memory = FakeMemory()
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
        min_judge_confidence=0.60,
    )
    loop._last_factual_report = {}
    loop.session_path = Path(tempfile.mkdtemp()) / "session.jsonl"
    loop._factual_debug_enabled = False
    loop._factual_debug_phase = None
    loop._factual_debug_path = loop.session_path.with_name(
        loop.session_path.stem + "_factual_debug.jsonl"
    )
    loop.output_guard = None
    loop.self_correction = None
    loop.conversation_state_manager = None
    loop._last_orchestration = {}
    loop._last_cognitive_state = None
    return loop


def test_1_real_retriever_contract_is_used():
    loop = make_loop()

    fake_retriever = types.SimpleNamespace(
        search=lambda query, top_k: [{
            "id": "doc-real-shape",
            "text": "Fragmento recuperado.",
            "score": 0.81,
            "meta": {"path": "docs/source.md", "chunk": 1},
        }]
    )
    fake_rag = types.ModuleType("rag")
    fake_rag.retriever = fake_retriever

    previous = sys.modules.get("rag")
    sys.modules["rag"] = fake_rag
    try:
        rows = loop._factual_evidence_provider("claim", 3)
    finally:
        if previous is None:
            sys.modules.pop("rag", None)
        else:
            sys.modules["rag"] = previous

    assert len(rows) == 1
    assert rows[0]["id"] == "doc-real-shape"
    assert rows[0]["meta"]["path"] == "docs/source.md"
    print("TEST 1 — QuipuLoop usa el contrato real rag.retriever.search()")


def test_2_judge_receives_only_claim_and_evidence_payload():
    loop = make_loop()

    claim = "El horizonte coincide con la singularidad."
    evidence = loop.factual_verifier.retrieve_evidence(
        loop.factual_verifier.extract_claims(claim, domain="physics")[0]
    )

    raw = loop._factual_judge_adapter(claim, evidence)
    parsed = json.loads(raw)

    assert parsed["status"] == "contradicted"
    call = loop.adapter.calls[-1]
    assert len(call) == 2
    payload = json.loads(call[-1]["content"])
    assert set(payload.keys()) == {"claim", "evidence"}
    assert payload["claim"] == claim
    assert "src_schwarzschild_1" in json.dumps(payload)
    assert "Aster" not in json.dumps(payload)
    print("TEST 2 — El juez recibe solo claim + evidencia grounded")


def test_3_contradiction_triggers_regeneration_and_second_pass():
    loop = make_loop()
    messages = [{"role": "user", "content": "¿Qué es un horizonte de sucesos?"}]

    answer = loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos?",
        "El horizonte de sucesos coincide con la singularidad.",
        messages,
    )

    assert "r = 2GM/c^2" in answer
    assert "r = 0" in answer
    assert "coincide con la singularidad" not in answer.lower()
    assert loop._last_factual_report["contradicted_count"] == 0
    print("TEST 3 — Contradicción => regeneración => segunda verificación")


def test_4_persistent_contradiction_is_not_exposed():
    loop = make_loop(adapter=AlwaysWrongAdapter())
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
    )

    answer = loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos?",
        "El horizonte de sucesos coincide con la singularidad.",
        [{"role": "user", "content": "¿Qué es un horizonte de sucesos?"}],
    )

    assert "coincide con la singularidad" not in answer.lower()
    assert "No pude producir una respuesta factual" in answer
    print("TEST 4 — Una contradicción persistente no llega al usuario")


def test_5_non_scientific_chat_does_not_trigger_judge():
    loop = make_loop()
    before = len(loop.adapter.calls)

    answer = loop._apply_factual_verification(
        "Hola, ¿cómo estás?",
        "Estoy bien, gracias.",
        [{"role": "user", "content": "Hola, ¿cómo estás?"}],
    )

    assert answer == "Estoy bien, gracias."
    assert len(loop.adapter.calls) == before
    assert loop._last_factual_report["metadata"]["skipped"] is True
    print("TEST 5 — Chat no científico no consulta RAG/juez")


def test_6_run_once_corrects_bad_scientific_draft_before_memory():
    loop = make_loop()

    loop._prepare_cognitive_state = lambda _text: None
    loop._prepare_orchestration = lambda _text: {
        "ok": True,
        "skill": "chat",
        "confidence": 0.55,
        "reason": "test",
        "output": None,
    }
    loop._build_messages = lambda user_text, orchestration=None, cognitive_state=None: [
        {"role": "user", "content": user_text}
    ]
    loop._metacognitive_block_answer = lambda _state: None
    loop._deterministic_quantitative_answer = lambda _text, _state: None
    loop._build_verifier_evidence = lambda: None
    loop._verify_answer = lambda answer, messages, evidence=None, cognitive_state=None: str(answer)
    loop._strip_internal_orchestration_leaks = lambda answer: str(answer)
    loop._apply_output_guard = lambda state, answer, messages=None, allow_regeneration=False: str(answer)
    loop._log_cognitive_state = lambda _state: None

    answer = loop.run_once("¿Qué es un horizonte de sucesos?")

    assert "r = 2GM/c^2" in answer
    assert "r = 0" in answer
    assert "coincide con la singularidad" not in answer.lower()
    assistant_rows = [row for row in loop.memory.saved if row[0] == "assistant"]
    assert assistant_rows
    assert assistant_rows[-1][1] == answer
    assert "coincide con la singularidad" not in assistant_rows[-1][1].lower()
    print("TEST 6 — run_once corrige antes de OutputGuard/memoria")



def _read_debug_rows(loop):
    path = Path(loop._factual_debug_path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_7_debug_trace_records_real_retrieval_judge_and_both_passes():
    loop = make_loop()
    loop._factual_debug_enabled = True

    fake_retriever = types.SimpleNamespace(
        search=lambda query, top_k: EVIDENCE[:top_k]
    )
    fake_rag = types.ModuleType("rag")
    fake_rag.retriever = fake_retriever

    previous = sys.modules.get("rag")
    sys.modules["rag"] = fake_rag
    try:
        loop.factual_verifier = FactualVerifier(
            evidence_provider=loop._factual_evidence_provider,
            judge_adapter=loop._factual_judge_adapter,
            min_judge_confidence=0.60,
        )
        answer = loop._apply_factual_verification(
            "¿Qué es un horizonte de sucesos?",
            "El horizonte de sucesos coincide con la singularidad.",
            [{"role": "user", "content": "¿Qué es un horizonte de sucesos?"}],
        )
    finally:
        if previous is None:
            sys.modules.pop("rag", None)
        else:
            sys.modules["rag"] = previous

    assert "r = 2GM/c^2" in answer

    rows = _read_debug_rows(loop)
    events = [row.get("event") for row in rows]
    phases = [row.get("phase") for row in rows]

    for expected in (
        "draft",
        "retrieval",
        "judge",
        "report",
        "corrected_before_classic_verifier",
        "corrected_after_classic_verifier",
        "final_decision",
    ):
        assert expected in events

    assert "initial" in phases
    assert "corrected" in phases

    judge_rows = [row for row in rows if row.get("event") == "judge"]
    assert judge_rows[0]["payload"]["claim"]
    assert judge_rows[0]["payload"]["evidence"]
    assert judge_rows[0]["payload"]["raw_judgement"]

    retrieval_rows = [row for row in rows if row.get("event") == "retrieval"]
    assert retrieval_rows[0]["payload"]["hits"][0]["id"] == "src_schwarzschild_1"

    print("TEST 7 — Debug registra retrieval + judge + report en ambas pasadas")


def test_8_debug_trace_reveals_persistent_contradiction():
    loop = make_loop(adapter=AlwaysWrongAdapter())
    loop._factual_debug_enabled = True
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
    )

    answer = loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos?",
        "El horizonte de sucesos coincide con la singularidad.",
        [{"role": "user", "content": "¿Qué es un horizonte de sucesos?"}],
    )

    assert "No pude producir una respuesta factual" in answer

    rows = _read_debug_rows(loop)
    reports = [row for row in rows if row.get("event") == "report"]
    finals = [row for row in rows if row.get("event") == "final_decision"]

    assert len(reports) >= 2
    assert reports[0]["phase"] == "initial"
    assert reports[-1]["phase"] == "corrected"
    assert reports[-1]["payload"]["contradicted_count"] >= 1
    assert finals[-1]["payload"]["accepted"] is False
    assert finals[-1]["payload"]["reason"] == "persistent_contradiction"

    print("TEST 8 — Debug muestra exactamente la contradicción persistente")


def test_9_false_contradiction_without_evidence_ids_is_downgraded():
    loop = make_loop(
        adapter=RealFailureReproductionAdapter()
    )
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
        min_judge_confidence=0.60,
    )

    answer = loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos y dónde se encuentra respecto "
        "a la singularidad en un agujero negro de Schwarzschild?",
        "El horizonte de sucesos coincide con la singularidad.",
        [{
            "role": "user",
            "content": "¿Qué es un horizonte de sucesos?"
        }],
    )

    assert "No pude producir una respuesta factual" not in answer
    assert "r = 0" in answer
    assert loop._last_factual_report["contradicted_count"] == 0
    assert loop._last_factual_report["uncertain_count"] >= 1

    print(
        "TEST 9 — Falso CONTRADICTED sin evidence_ids => UNCERTAIN, no bloqueo"
    )


def test_10_factual_retrieval_filters_off_topic_noise():
    loop = make_loop()

    rows = [
        {
            "id": "bh-1",
            "text": "Horizonte y singularidad son regiones distintas.",
            "score": 0.74,
            "meta": {"path": "science/black_holes.md", "chunk": 1},
        },
        {
            "id": "bh-2",
            "text": "El horizonte está en r_s = 2GM/c^2.",
            "score": 0.56,
            "meta": {"path": "science/black_holes.md", "chunk": 2},
        },
        {
            "id": "geo-1",
            "text": "Las placas tectónicas se desplazan.",
            "score": 0.34,
            "meta": {"path": "science/geology.md", "chunk": 0},
        },
        {
            "id": "water-1",
            "text": "El agua tiene fórmula H2O.",
            "score": 0.29,
            "meta": {"path": "science/water.md", "chunk": 0},
        },
    ]

    fake_retriever = types.SimpleNamespace(
        search=lambda query, top_k: rows[:top_k]
    )
    fake_rag = types.ModuleType("rag")
    fake_rag.retriever = fake_retriever

    previous = sys.modules.get("rag")
    sys.modules["rag"] = fake_rag
    try:
        filtered = loop._factual_evidence_provider(
            "horizonte de sucesos Schwarzschild",
            5,
        )
    finally:
        if previous is None:
            sys.modules.pop("rag", None)
        else:
            sys.modules["rag"] = previous

    ids = [row["id"] for row in filtered]
    assert ids == ["bh-1", "bh-2"]

    print("TEST 10 — Factual RAG elimina ruido off-topic débil")


def test_11_factual_block_is_not_reused_as_conversation_context():
    loop = make_loop(
        adapter=AlwaysWrongAdapter()
    )
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
    )

    loop._prepare_cognitive_state = lambda _text: None
    loop._prepare_orchestration = lambda _text: {
        "ok": True,
        "skill": "chat",
        "confidence": 0.55,
        "reason": "test",
        "output": None,
    }
    loop._build_messages = lambda user_text, orchestration=None, cognitive_state=None: [
        {"role": "user", "content": user_text}
    ]
    loop._metacognitive_block_answer = lambda _state: None
    loop._deterministic_quantitative_answer = lambda _text, _state: None
    loop._build_verifier_evidence = lambda: None
    loop._verify_answer = lambda answer, messages, evidence=None, cognitive_state=None: str(answer)
    loop._strip_internal_orchestration_leaks = lambda answer: str(answer)
    loop._apply_output_guard = lambda state, answer, messages=None, allow_regeneration=False: str(answer)
    loop._log_cognitive_state = lambda _state: None

    answer = loop.run_once(
        "¿Qué es un horizonte de sucesos?"
    )

    assert "No pude producir una respuesta factual" in answer

    assistant_rows = [
        row
        for row in loop.memory.saved
        if row[0] == "assistant"
    ]
    assert assistant_rows

    metadata = assistant_rows[-1][2]
    assert metadata["context_reuse"] is False
    assert metadata["cognitive_status"] == "factual_verification_blocked"
    assert metadata["factual_verifier_status"] == "persistent_contradiction"

    print("TEST 11 — Bloqueo factual se guarda con context_reuse=False")


def test_12_valid_id_false_contradiction_does_not_block_corrected_answer():
    loop = make_loop(
        adapter=ValidIdFalseContradictionAdapter()
    )
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
        min_judge_confidence=0.60,
    )

    answer = loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos y dónde se encuentra respecto "
        "a la singularidad en un agujero negro de Schwarzschild?",
        "El horizonte de sucesos coincide con la singularidad.",
        [{
            "role": "user",
            "content": (
                "¿Qué es un horizonte de sucesos y dónde se encuentra "
                "respecto a la singularidad?"
            ),
        }],
    )

    assert "No pude producir una respuesta factual" not in answer
    assert "r = 2GM/c^2" in answer
    assert "r = 0" in answer
    assert "regiones distintas" in answer
    assert loop._last_factual_report["contradicted_count"] == 0
    assert loop._last_factual_report["uncertain_count"] >= 1

    verdicts = loop._last_factual_report["verdicts"]
    downgraded = [
        item
        for item in verdicts
        if item.get("status") == "uncertain"
        and item.get("contradiction_validation")
    ]
    assert downgraded
    assert downgraded[0]["contradiction_validation"]["valid"] is False

    print(
        "TEST 12 — CONTRADICTED + ID válido pero evidencia compatible "
        "=> UNCERTAIN, respuesta corregida pasa"
    )


def test_13_real_explicit_contradiction_still_blocks():
    loop = make_loop()

    report = loop.factual_verifier.verify(
        "El horizonte de sucesos coincide con la singularidad.",
        user_text="Explícame Schwarzschild.",
        force=True,
    )

    assert report.contradicted_count == 1
    assert report.correction_required is True
    assert report.verdicts[0].contradiction_validation["valid"] is True

    print("TEST 13 — Contradicción explícita real sigue bloqueando")


def test_14_judge_contract_requests_exact_contradiction_fields():
    loop = make_loop()

    claim = "El horizonte coincide con la singularidad."
    evidence = loop.factual_verifier.retrieve_evidence(
        loop.factual_verifier.extract_claims(
            claim,
            domain="physics",
        )[0]
    )

    loop._factual_judge_adapter(
        claim,
        evidence,
    )

    system = loop.adapter.calls[-1][0]["content"]

    for field in (
        "claim_fact",
        "evidence_fact",
        "evidence_quote",
        "contradiction_type",
    ):
        assert field in system

    print("TEST 14 — Juez recibe contrato de contradicción verificable")


def test_15_debug_report_exposes_contradiction_validation():
    loop = make_loop(
        adapter=ValidIdFalseContradictionAdapter()
    )
    loop._factual_debug_enabled = True
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
    )

    loop._apply_factual_verification(
        "¿Qué es un horizonte de sucesos?",
        "El horizonte de sucesos coincide con la singularidad.",
        [{"role": "user", "content": "¿Qué es un horizonte de sucesos?"}],
    )

    rows = _read_debug_rows(loop)
    reports = [
        row
        for row in rows
        if row.get("event") == "report"
    ]
    assert reports

    found = any(
        verdict.get("contradiction_validation")
        for report in reports
        for verdict in report["payload"].get("verdicts", [])
    )
    assert found

    print("TEST 15 — Debug expone ContradictionValidator")


def test_16_fabricated_quote_is_rejected_even_with_valid_id():
    class FabricatedQuoteAdapter(FakeAdapter):
        def generate(self, messages, **kwargs):
            rendered = "\n".join(
                str(item.get("content") or "")
                for item in messages
            )
            self.calls.append(list(messages))

            if "juez factual basado estrictamente en evidencia" in rendered:
                return json.dumps({
                    "status": "contradicted",
                    "confidence": 0.99,
                    "reason": "Cita inventada.",
                    "evidence_ids": ["src_schwarzschild_1"],
                    "evidence_quote": "El horizonte está dentro de la singularidad.",
                    "claim_fact": "El horizonte coincide con la singularidad.",
                    "evidence_fact": "El horizonte está dentro de la singularidad.",
                    "contradiction_type": "identity_conflict",
                })

            return "El horizonte coincide con la singularidad."

    loop = make_loop(
        adapter=FabricatedQuoteAdapter()
    )
    loop.factual_verifier = FactualVerifier(
        evidence_provider=evidence_provider,
        judge_adapter=loop._factual_judge_adapter,
    )

    report = loop.factual_verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame Schwarzschild.",
        force=True,
    )

    assert report.contradicted_count == 0
    assert report.uncertain_count == 1
    validation = report.verdicts[0].contradiction_validation
    assert validation["contradiction_type"] == "quote_not_found"

    print("TEST 16 — Cita inventada no puede producir bloqueo")

def main():
    test_1_real_retriever_contract_is_used()
    test_2_judge_receives_only_claim_and_evidence_payload()
    test_3_contradiction_triggers_regeneration_and_second_pass()
    test_4_persistent_contradiction_is_not_exposed()
    test_5_non_scientific_chat_does_not_trigger_judge()
    test_6_run_once_corrects_bad_scientific_draft_before_memory()
    test_7_debug_trace_records_real_retrieval_judge_and_both_passes()
    test_8_debug_trace_reveals_persistent_contradiction()
    test_9_false_contradiction_without_evidence_ids_is_downgraded()
    test_10_factual_retrieval_filters_off_topic_noise()
    test_11_factual_block_is_not_reused_as_conversation_context()
    test_12_valid_id_false_contradiction_does_not_block_corrected_answer()
    test_13_real_explicit_contradiction_still_blocks()
    test_14_judge_contract_requests_exact_contradiction_fields()
    test_15_debug_report_exposes_contradiction_validation()
    test_16_fabricated_quote_is_rejected_even_with_valid_id()

    print("=" * 100)
    print("FACTUAL VERIFIER CONTRADICTION VALIDATION 1.0.4: 16/16 pruebas superadas.")


if __name__ == "__main__":
    main()
