"""Regresión del fallo real Asterion/Helios; sin Qwen ni memoria personal."""
from types import SimpleNamespace
from quipu_loop import QuipuLoop

ASTERION = (
    "Durante esta conversación, recuerda que mi proyecto ficticio se llama "
    "Proyecto Asterion, tiene tres módulos llamados Órbita, Prisma y Centinela, "
    "y su objetivo es estudiar las atmósferas de exoplanetas."
)
HELIOS = (
    "Durante esta conversación, recuerda que mi proyecto ficticio se llama "
    "Helios, tiene tres módulos llamados Atlas, Nexo y Prisma, "
    "y su objetivo es estudiar exoplanetas."
)
QUERY = (
    "¿Cómo se llama el proyecto ficticio que te mencioné, cuáles son sus tres "
    "módulos y cuál es su objetivo?"
)

def make_loop():
    loop=object.__new__(QuipuLoop)
    loop._conversation_turns_by_session={}
    loop._run_once_impl=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Llamó al LLM'))
    return loop

def test_structured_declaration_preserves_exact_user_data():
    loop=make_loop()
    answer=loop.run_once(ASTERION)
    assert "Entendido" in answer
    assert loop._conversation_turns_by_session['__cli__'][0]['content']==ASTERION

def test_explicit_session_recall_does_not_hallucinate():
    loop=make_loop()
    loop.run_once(ASTERION)
    answer=loop.run_once(QUERY)
    assert all(word in answer for word in ('Proyecto Asterion','Órbita','Prisma','Centinela','atmósferas de exoplanetas'))
    assert 'Ecosistema Digital' not in answer and 'Insignia' not in answer

def test_other_chat_cannot_receive_project():
    loop=make_loop()
    loop.run_once(ASTERION,session_id='chat-a')
    loop.run_once(HELIOS,session_id='chat-b')
    b=loop.run_once(QUERY,session_id='chat-b')
    a=loop.run_once(QUERY,session_id='chat-a')
    assert 'Helios' in b and 'Asterion' not in b
    assert 'Proyecto Asterion' in a and 'Helios' not in a

def test_http_override_is_authoritative():
    loop=make_loop()
    loop.run_once(HELIOS,session_id='chat-a')
    a=loop.run_once(QUERY,session_id='chat-a',recent_context_override=[{'role':'user','content':ASTERION}])
    assert 'Asterion' in a and 'Helios' not in a

def test_blocked_user_override_not_reused():
    loop=make_loop()
    result=loop._direct_session_memory_response(
        QUERY,'chat-z',[{'role':'user','content':ASTERION,'metadata':{'context_reuse':False}}]
    )
    assert result is None

def test_no_global_memory_fallback_when_chat_has_no_declaration():
    loop=make_loop()
    assert loop._direct_session_memory_response(QUERY,'chat-z',None) is None

def test_unrelated_question_uses_normal_model_route():
    loop=make_loop()
    assert loop._direct_session_memory_response('Explícame la evolución de las estrellas',None,None) is None

def test_general_memory_writes_still_use_normal_route():
    loop=make_loop()
    assert loop._direct_session_memory_response('Recuerda que debo trabajar',None,None) is None

def test_direct_memory_does_not_write_global_persistent_memory():
    loop=make_loop()
    loop.memory=SimpleNamespace(add=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('No guardar globalmente')))
    loop.run_once(ASTERION)
    loop.run_once(QUERY)
