from __future__ import annotations

from verifier.factual_verifier import FactualVerifier


EVIDENCE = [{
    "id": "src-1",
    "text": (
        "---\n"
        "curation_authority_score: 0.98\n"
        "---\n"
        "El horizonte está en r_s. La singularidad clásica está en r = 0. "
        "Son regiones distintas."
    ),
    "score": 0.80,
    "meta": {"path": "science/black_holes.md"},
}, {
    "id": "src-2",
    "text": "El horizonte está en el radio de Schwarzschild.",
    "score": 0.70,
    "meta": {"path": "science/black_holes.md"},
}]


def provider(_claim, top_k):
    return EVIDENCE[:top_k]


def false_contradiction_without_ids(_claim, _evidence):
    return {
        "status": "contradicted",
        "confidence": 0.98,
        "reason": "Creo que hay una contradicción, pero no cito el fragmento.",
    }


def explicit_contradiction_with_ids(_claim, _evidence):
    return {
        "status": "contradicted",
        "confidence": 0.98,
        "reason": "La evidencia dice que son regiones distintas.",
        "evidence_ids": ["src-1"],
    }


def test_missing_ids_downgrades_contradiction():
    verifier = FactualVerifier(
        provider,
        false_contradiction_without_ids,
    )
    report = verifier.verify(
        "El horizonte no está relacionado directamente con la singularidad.",
        user_text="Explícame un agujero negro.",
        force=True,
    )

    assert report.contradicted_count == 0
    assert report.uncertain_count == 1
    assert report.correction_required is False
    assert report.safe_to_answer is True

    print("TEST 1 — CONTRADICTED sin evidence_ids se degrada a UNCERTAIN")


def test_explicit_ids_preserve_real_contradiction():
    verifier = FactualVerifier(
        provider,
        explicit_contradiction_with_ids,
    )
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame un agujero negro.",
        force=True,
    )

    assert report.contradicted_count == 1
    assert report.correction_required is True
    assert report.safe_to_answer is False

    print("TEST 2 — CONTRADICTED con evidencia explícita sigue bloqueando")


def test_curated_authority_propagates_to_sibling_chunks():
    verifier = FactualVerifier(provider, None)
    claim = verifier.extract_claims(
        "El horizonte está en el radio de Schwarzschild.",
        domain="astronomy",
    )[0]

    evidence = verifier.retrieve_evidence(claim)

    assert len(evidence) == 2
    assert evidence[0].authority == 0.98
    assert evidence[1].authority == 0.98

    print("TEST 3 — Autoridad curada se propaga entre chunks del mismo documento")



def false_contradiction_with_valid_id_but_supporting_evidence(_claim, _evidence):
    return {
        "status": "contradicted",
        "confidence": 0.98,
        "reason": "Interpreto erróneamente separación como contradicción.",
        "evidence_ids": ["src-1"],
        "evidence_quote": "Son regiones distintas.",
        "claim_fact": "El horizonte y la singularidad son regiones distintas.",
        "evidence_fact": "Son regiones distintas.",
        "contradiction_type": "identity_conflict",
    }


def fabricated_quote_contradiction(_claim, _evidence):
    return {
        "status": "contradicted",
        "confidence": 0.99,
        "reason": "Cito una frase que no existe.",
        "evidence_ids": ["src-1"],
        "evidence_quote": "El horizonte está dentro de la singularidad.",
        "claim_fact": "El horizonte coincide con la singularidad.",
        "evidence_fact": "El horizonte está dentro de la singularidad.",
        "contradiction_type": "identity_conflict",
    }


def real_contradiction_with_exact_quote(_claim, _evidence):
    return {
        "status": "contradicted",
        "confidence": 0.99,
        "reason": "La evidencia niega explícitamente la identidad.",
        "evidence_ids": ["src-1"],
        "evidence_quote": "Son regiones distintas.",
        "claim_fact": "El horizonte coincide con la singularidad.",
        "evidence_fact": "Son regiones distintas.",
        "contradiction_type": "identity_conflict",
    }


def test_valid_id_is_not_enough_when_evidence_supports_claim():
    verifier = FactualVerifier(
        provider,
        false_contradiction_with_valid_id_but_supporting_evidence,
    )
    report = verifier.verify(
        "El horizonte y la singularidad son regiones distintas.",
        user_text="Explícame Schwarzschild.",
        force=True,
    )

    assert report.contradicted_count == 0
    assert report.uncertain_count == 1
    assert report.correction_required is False
    assert report.verdicts[0].contradiction_validation["valid"] is False

    print("TEST 4 — evidence_id válido no basta si evidencia apoya la claim")


def test_fabricated_evidence_quote_cannot_block():
    verifier = FactualVerifier(
        provider,
        fabricated_quote_contradiction,
    )
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame Schwarzschild.",
        force=True,
    )

    assert report.contradicted_count == 0
    assert report.uncertain_count == 1
    validation = report.verdicts[0].contradiction_validation
    assert validation["valid"] is False
    assert validation["contradiction_type"] == "quote_not_found"

    print("TEST 5 — evidence_quote inventada => no bloqueo")


def test_real_identity_conflict_with_exact_quote_is_preserved():
    verifier = FactualVerifier(
        provider,
        real_contradiction_with_exact_quote,
    )
    report = verifier.verify(
        "El horizonte coincide con la singularidad.",
        user_text="Explícame Schwarzschild.",
        force=True,
    )

    assert report.contradicted_count == 1
    assert report.correction_required is True
    validation = report.verdicts[0].contradiction_validation
    assert validation["valid"] is True
    assert validation["contradiction_type"] == "identity_conflict"

    print("TEST 6 — Identidad vs regiones distintas => contradicción validada")

def main():
    test_missing_ids_downgrades_contradiction()
    test_explicit_ids_preserve_real_contradiction()
    test_curated_authority_propagates_to_sibling_chunks()
    test_valid_id_is_not_enough_when_evidence_supports_claim()
    test_fabricated_evidence_quote_cannot_block()
    test_real_identity_conflict_with_exact_quote_is_preserved()

    print("=" * 100)
    print("FACTUAL VERIFIER REGRESSION 1.0.4: 6/6 pruebas superadas.")


if __name__ == "__main__":
    main()
