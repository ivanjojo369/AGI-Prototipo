# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import List

try:
    from root.quipu_loop import QuipuLoop
except Exception:
    from quipu_loop import QuipuLoop  # type: ignore

try:
    from root.settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT
except Exception:
    try:
        from settings import RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT  # type: ignore
    except Exception:
        RAG_SCORE_THRESHOLD, RAG_TOPK_DEFAULT = 0.35, 5

from memory.memory import write as mem_write, search as mem_search, prune as mem_prune, reindex as mem_reindex
from rag.retriever import search as rag_search

def _exists_dir(p: str|Path) -> bool:
    p = Path(p); return p.exists() and p.is_dir()

def reindex(paths: List[str], *, fresh_first: bool=True) -> bool:
    ok=True; first=False
    for p in paths:
        if not _exists_dir(p): continue
        mode = "fresh" if (fresh_first and not first) else "append"
        first = True or first
        cmd=[sys.executable,"-m","scripts.index_folder","--path",p,"--ext",".py,.md,.txt","--mode",mode]
        print("  "," ".join(cmd))
        proc=subprocess.run(cmd,capture_output=True,text=True)
        if proc.returncode!=0: print("[index_folder] ERROR:",proc.stderr.strip()); ok=False
        else: print(proc.stdout.strip())
    return ok

def test_memory()->bool:
    print(" Test Memoria")
    r1=mem_write("nota e2e: preferencias RAG vs heurísticas",user="tester",project_id="e2e",tags=["e2e"])
    if not r1.get("ok"): print("   write"); return False
    if not mem_search("preferencias RAG",topk=3,project_id="e2e"): print("   search"); return False
    if not mem_prune().get("ok"): print("   prune"); return False
    if not mem_reindex(project_id="e2e").get("ok"): print("   reindex"); return False
    print("   Memoria OK"); return True

def test_rag(min_score:float)->bool:
    print(" Test RAG")
    for q in ["memoria episodica","episodios de memoria","settings del proyecto","retriever rag","loop quipu"]:
        if rag_search(q,top_k=5,min_score=min_score):
            print(f"   RAG OK con '{q}' (min_score={min_score})"); return True
    print("   RAG sin resultados"); return False

def test_loop(min_score:float,top_k:int)->bool:
    print(" Test Loop")
    out=QuipuLoop(project_id='e2e',min_score=min_score,top_k=top_k).run("define episodios de memoria y su estructura")
    ok=bool(out.get("ok")) and isinstance(out.get("output"),str)
    print("  salida:",(out.get("output") or "")[:200].replace("\n"," "),"")
    print("  stats:",out.get("stats")); print("  verified:",out.get("verified"))
    return ok and "ok" in out.get("verified",{})

def main():
    ap=argparse.ArgumentParser("run_e2e_test")
    ap.add_argument("--min-score",type=float,default=RAG_SCORE_THRESHOLD)
    ap.add_argument("--top-k",type=int,default=RAG_TOPK_DEFAULT)
    ap.add_argument("--paths",nargs="*",default=["root","memory","rag","executive","docs"])
    a=ap.parse_args()

    print("== Reindexar =="); idx_ok=reindex(a.paths,fresh_first=True)
    print("\n== Pruebas ==")
    m_ok=test_memory(); r_ok=test_rag(a.min_score); l_ok=test_loop(a.min_score,a.top_k)
    summary={"index_ok":idx_ok,"memory_ok":m_ok,"rag_ok":r_ok,"loop_ok":l_ok}
    print("\n== RESUMEN =="); print(json.dumps(summary,ensure_ascii=False,indent=2))
    sys.exit(0 if all(summary.values()) else 1)

if __name__=="__main__": main()
