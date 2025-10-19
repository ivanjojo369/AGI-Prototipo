# tests/test_memory_forgetting.py
import time
from memory.unified_memory import UnifiedMemory

def test_ttl_and_decay():
    m = UnifiedMemory(storage_dir="./sessions/testmem", ttl_days=0, half_life_days=1, max_items=100)
    k = m.upsert("texto viejo", {"id": "old"})
    # TTL 0 días → purge_ttl lo debe borrar
    m.purge_ttl()
    assert k not in m.items

    # Decaimiento
    k2 = m.upsert("texto reciente", {"id": "new"})
    it = m.items[k2]
    it.last_access -= 2*24*3600  # 2 días sin acceso
    old_score = it.score
    m.decay_scores()
    assert it.score < old_score
