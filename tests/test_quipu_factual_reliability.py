"""Quipu Factual Reliability RC 1.1: run with python -m unittest discover -s tests -p test_quipu_factual_reliability.py -v"""
import unittest
import ast
from pathlib import Path
from unittest.mock import patch

from verifier.support_validator import SupportValidator
from verifier.factual_verifier import FactualVerifier, EvidenceItem


class TestSupportValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SupportValidator()

    def sample(self, claim, source, quote=None, fact=None):
        return self.validator.validate(claim, [{"evidence_id": "e1", "text": source}], {
            "evidence_ids": ["e1"], "claim_fact": claim,
            "evidence_fact": fact or source,
            "evidence_quote": source if quote is None else quote,
        })

    def test_exact_claim_is_supported(self):
        text = "La velocidad de la luz es 299792 km/s."
        self.assertTrue(self.sample(text, text).valid)

    def test_number_mismatch_is_not_supported(self):
        self.assertFalse(self.sample(
            "La velocidad de la luz es 300000 km/s.",
            "La velocidad de la luz es 3 km/s.").valid)

    def test_negation_mismatch_is_not_supported(self):
        self.assertFalse(self.sample(
            "El organismo pertenece a los hongos.",
            "El organismo no pertenece a los hongos.").valid)

    def test_fabricated_quote_rejected(self):
        result = self.sample(
            "El fósil pertenece a los hongos.",
            "El fósil pertenece a los hongos.",
            quote="El fósil es un animal.")
        self.assertFalse(result.valid)

    def test_unrelated_negation_elsewhere_in_source_not_mistaken_for_contradiction(self):
        claim = "El horizonte de sucesos está en r = 2GM/c^2."
        source = (
            "Para un agujero negro de Schwarzschild, el horizonte de sucesos "
            "está en r = 2GM/c^2. La singularidad clásica está en r = 0; "
            "no son la misma ubicación."
        )
        result = self.validator.validate(
            claim, [{"evidence_id": "bh", "text": source}],
            {"evidence_ids": ["bh"], "claim_fact": claim}
        )
        self.assertTrue(result.valid, result.reason)


class TestFactualVerifier(unittest.TestCase):
    def test_fossil_question_triggers_verification(self):
        self.assertTrue(FactualVerifier.should_verify(
            "¿A qué reino pertenece este fósil?", "Pertenece a los hongos."))

    def test_fossil_claim_extracted(self):
        verifier = FactualVerifier()
        claims = verifier.extract_claims("El fósil pertenece al reino de los hongos.")
        self.assertGreaterEqual(len(claims), 1)

    def test_unverified_claim_requires_caveat(self):
        verifier = FactualVerifier(evidence_provider=lambda *_: [])
        report = verifier.verify(
            "El fósil pertenece al reino de los hongos.",
            user_text="Clasifica este fósil.")
        self.assertTrue(report.requires_caveat)
        self.assertEqual(report.unverifiable_count, 1)
        self.assertFalse(report.correction_required)

    def test_invalid_numeric_support_forces_regeneration(self):
        def fake_judge(*_):
            return {"status": "supported", "confidence": 0.99,
                "reason": "ejemplo", "evidence_ids": ["e1"],
                "claim_fact": "La velocidad de la luz es 300000 km/s.",
                "evidence_fact": "La velocidad de la luz es 3 km/s.",
                "evidence_quote": "La velocidad de la luz es 3 km/s."}
        verifier = FactualVerifier(
            evidence_provider=lambda *_: [{"evidence_id": "e1", "text": "La velocidad de la luz es 3 km/s."}],
            judge_adapter=fake_judge)
        report = verifier.verify("La velocidad de la luz es 300000 km/s.", user_text="física")
        self.assertEqual(report.support_failure_count, 1)
        self.assertTrue(report.correction_required)


class TestRetriever(unittest.TestCase):
    def test_factual_search_does_not_truncate(self):
        from rag import retriever
        long_text = "Fósil " * 350 + "EVIDENCIA_FINAL_LITERAL"
        index = [{"id": "d1", "vec": [1.0, 0.0], "text": long_text, "meta": {"path": "prueba"}}]
        with patch.object(retriever, "_load_index", return_value=index), \
             patch.object(retriever, "embed_text", return_value=[1.0, 0.0]):
            full = retriever.search("fósil", max_chunk_len=0)
            short = retriever.search("fósil")
        self.assertIn("EVIDENCIA_FINAL_LITERAL", full[0]["text"])
        self.assertNotIn("EVIDENCIA_FINAL_LITERAL", short[0]["text"])


class TestLoopIntegrationContract(unittest.TestCase):
    def test_caveat_function_is_conservative(self):
        source = Path(__file__).resolve().parents[1] / "quipu_loop.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "QuipuLoop")
        helper = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "_factual_with_caveat")
        standalone = ast.Module(body=[helper], type_ignores=[])
        # Strip the @staticmethod wrapper when compiled outside the class.
        helper.decorator_list = []
        ast.fix_missing_locations(standalone)
        scope = {"Any": object}
        exec(compile(standalone, str(source), "exec"), scope)
        fn = scope["_factual_with_caveat"]
        class Report:
            requires_caveat = True
        self.assertIn("no pudieron confirmarse", fn("Ejemplo", Report()))
        self.assertEqual("Ejemplo", fn("Ejemplo", type("R", (), {"requires_caveat": False})()))
        self.assertEqual("", fn("", Report()))

    def test_factual_search_untruncated_contract(self):
        import sys
        import types
        from unittest.mock import patch
        from quipu_loop import QuipuLoop

        observed = {}
        def search(query, top_k, max_chunk_len=1200):
            observed.update(query=query, top_k=top_k, max_chunk_len=max_chunk_len)
            return [{"id": "test-doc", "text": "Referencia íntegra", "score": 0.9,
                     "meta": {"path": "test.md"}}]

        fake_rag = types.ModuleType("rag")
        fake_rag.retriever = types.SimpleNamespace(search=search)
        loop = object.__new__(QuipuLoop)
        loop.cfg = types.SimpleNamespace(use_rag=True)
        loop._factual_debug_enabled = False
        with patch.dict(sys.modules, {"rag": fake_rag}):
            rows = loop._factual_evidence_provider("referencia", 3)
        self.assertEqual(0, observed["max_chunk_len"])
        self.assertEqual("test-doc", rows[0]["id"])

    def test_factual_search_legacy_signature_supported(self):
        import sys
        import types
        from unittest.mock import patch
        from quipu_loop import QuipuLoop

        fake_rag = types.ModuleType("rag")
        fake_rag.retriever = types.SimpleNamespace(
            search=lambda query, top_k: [{
                "id": "legacy", "text": "Referencia legada", "score": 0.85,
                "meta": {"path": "legacy.md"}}]
        )
        loop = object.__new__(QuipuLoop)
        loop.cfg = types.SimpleNamespace(use_rag=True)
        loop._factual_debug_enabled = False
        with patch.dict(sys.modules, {"rag": fake_rag}):
            rows = loop._factual_evidence_provider("referencia", 3)
        self.assertEqual("legacy", rows[0]["id"])


if __name__ == "__main__":
    unittest.main()
