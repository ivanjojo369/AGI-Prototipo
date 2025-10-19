# -*- coding: utf-8 -*-
from __future__ import annotations

from memory.memory import write, search, reindex, prune

def main():
    print(">> write")
    print(write("Preferencia: priorizo RAG sobre heuristicas.", user="ivan", project_id="Synchronexis", tags=["prefs"]))
    print(">> search")
    print(search("priorizo RAG", topk=3, project_id="Synchronexis"))
    print(">> prune")
    print(prune())
    print(">> reindex")
    print(reindex(project_id=None))

if __name__ == "__main__":
    main()
