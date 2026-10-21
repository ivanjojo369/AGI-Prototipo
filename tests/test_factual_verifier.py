from __future__ import annotations
from verifier.factual_verifier import FactualVerifier

EVIDENCE = {
    "horizonte": [{"id":"src_schwarzschild_1","source":"manual_fisica","score":0.95,"authority":0.95,
        "text":"Para un agujero negro de Schwarzschild, el horizonte de sucesos está en r = 2GM/c^2. La singularidad clásica está en r = 0; no son la misma ubicación."}],
    "tierra": [{"id":"src_astronomia_1","source":"manual_astronomia","score":0.95,"authority":0.95,
        "text":"La Tierra orbita alrededor del Sol."}],
}

def provider(claim, top_k):
    t=claim.lower()
    if "horizonte" in t or "singularidad" in t: return EVIDENCE["horizonte"][:top_k]
    if "tierra" in t or "sol" in t: return EVIDENCE["tierra"][:top_k]
    return []

def grounded_judge(claim, evidence):
    t=claim.lower(); joined=" ".join(x.text.lower() for x in evidence)
    if "horizonte" in t and "singularidad" in t and "coincide" in t and "no son la misma" in joined:
        return {"status":"contradicted","confidence":0.99,"reason":"La evidencia los sitúa en ubicaciones distintas.","evidence_ids":["src_schwarzschild_1"]}
    if "horizonte" in t and "2gm/c^2" in t.replace(" ", "") and "r = 2gm/c^2" in joined:
        return {"status":"supported","confidence":0.98,"reason":"La evidencia respalda la ubicación del horizonte.","evidence_ids":["src_schwarzschild_1"]}
    if "tierra orbita alrededor del sol" in t and "tierra orbita alrededor del sol" in joined:
        return {"status":"supported","confidence":0.99,"reason":"Respaldado.","evidence_ids":["src_astronomia_1"]}
    return {"status":"uncertain","confidence":0.50,"reason":"Evidencia insuficiente.","evidence_ids":[x.evidence_id for x in evidence]}

def make_verifier(): return FactualVerifier(provider, grounded_judge, min_judge_confidence=0.60)

def test_1():
    r=make_verifier().verify("El horizonte de sucesos coincide con la singularidad.", user_text="¿Qué es un horizonte de sucesos?", force=True)
    assert r.contradicted_count==1 and r.correction_required and not r.safe_to_answer
    print("TEST 1 — Horizonte = singularidad => CONTRADICTED")

def test_2():
    r=make_verifier().verify("En Schwarzschild, el horizonte está en r = 2GM/c^2.", user_text="Explícame el horizonte de sucesos.", force=True)
    assert r.supported_count==1 and r.safe_to_answer
    print("TEST 2 — Horizonte en r=2GM/c^2 => SUPPORTED")

def test_3():
    r=make_verifier().verify("En Schwarzschild, el horizonte está en r = 2GM/c^2. El horizonte de sucesos coincide con la singularidad.", user_text="Háblame de agujeros negros.", force=True)
    assert r.supported_count==1 and r.contradicted_count==1 and r.correction_required
    print("TEST 3 — Respuesta mixta exige corrección")

def test_4():
    r=make_verifier().verify("La partícula XZ-99 tiene una masa exactamente igual a 14.7 unidades.", user_text="Explícame esta partícula.", force=True)
    assert r.unverifiable_count==1 and r.contradicted_count==0 and r.requires_caveat
    print("TEST 4 — Sin evidencia => UNVERIFIABLE")

def test_5():
    v=make_verifier()
    bad=v.verify("El horizonte de sucesos coincide con la singularidad.", user_text="¿Qué es un horizonte de sucesos?", force=True)
    assert bad.correction_required
    good=v.verify("En Schwarzschild, el horizonte está en r = 2GM/c^2. La singularidad clásica está en r = 0.", user_text="¿Qué es un horizonte de sucesos?", force=True)
    assert good.contradicted_count==0 and good.safe_to_answer
    print("TEST 5 — Respuesta corregida ya no contiene contradicción")

def test_6():
    r=make_verifier().verify("Estoy bien, gracias.", user_text="Hola, ¿cómo estás?")
    assert r.metadata.get("skipped") and r.metadata.get("skip_reason")=="no_science_signal"
    print("TEST 6 — Chat normal no activa verificación factual")



def test_7_retriever_meta_path_becomes_source():
    def rag_provider(_claim, _top_k):
        return [{
            "id": "doc-1",
            "text": "Evidencia científica de prueba.",
            "score": 0.91,
            "meta": {
                "path": "docs/fisica.md",
                "chunk": 3,
                "authority": 0.88,
            },
        }]

    v = FactualVerifier(rag_provider, None)
    claim = v.extract_claims(
        "La Tierra orbita alrededor del Sol."
    )[0]
    evidence = v.retrieve_evidence(claim)

    assert len(evidence) == 1
    assert evidence[0].source == "docs/fisica.md"
    assert evidence[0].metadata["chunk"] == 3
    assert evidence[0].authority == 0.88
    print("TEST 7 — meta.path del retriever se conserva como source")


def test_8_missing_authority_defaults_to_half():
    def rag_provider(_claim, _top_k):
        return [{
            "id": "doc-2",
            "text": "La Tierra orbita alrededor del Sol.",
            "score": 0.97,
            "meta": {"path": "docs/astronomia.md"},
        }]

    v = FactualVerifier(rag_provider, None)
    claim = v.extract_claims(
        "La Tierra orbita alrededor del Sol."
    )[0]
    evidence = v.retrieve_evidence(claim)

    assert evidence[0].authority == 0.5
    print("TEST 8 — authority ausente => 0.5 conservador")


def test_9_retrieval_score_is_not_authority():
    def rag_provider(_claim, _top_k):
        return [{
            "id": "doc-3",
            "text": "La Tierra orbita alrededor del Sol.",
            "score": 0.999,
            "meta": {"path": "docs/astronomia.md"},
        }]

    v = FactualVerifier(rag_provider, None)
    claim = v.extract_claims(
        "La Tierra orbita alrededor del Sol."
    )[0]
    evidence = v.retrieve_evidence(claim)

    assert evidence[0].score == 0.999
    assert evidence[0].authority == 0.5
    assert evidence[0].authority != evidence[0].score
    print("TEST 9 — retrieval score nunca se confunde con autoridad factual")

def main():
    for f in (
        test_1, test_2, test_3, test_4, test_5, test_6,
        test_7_retriever_meta_path_becomes_source,
        test_8_missing_authority_defaults_to_half,
        test_9_retrieval_score_is_not_authority,
    ):
        f()
    print("FACTUAL VERIFIER 1.0.1: 9/9 pruebas superadas.")
if __name__=='__main__': main()
