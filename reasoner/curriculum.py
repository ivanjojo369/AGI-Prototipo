# reasoner/curriculum.py
from __future__ import annotations

import glob
import csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MicroTask:
    """
    Tarea de práctica simple que el meta-agente puede intentar resolver.
    """
    goal: str
    hint: str = ""
    kind: str = "memory"  # "memory" | "math" | "file" | "web"
    expected_contains: Optional[str] = None  # opcional para validación blanda

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CurriculumBuilder:
    """
    Construye un set de micro-tareas a partir de los CSVs de benchmarks
    que ya produces (p. ej., memory_metrics_YYYYMMDD.csv).
    La heurística busca filas con señales de fallo y genera prompts de práctica.
    """

    def __init__(self) -> None:
        self.tasks: List[MicroTask] = []

    def _looks_like_failure(self, row: Dict[str, str]) -> bool:
        """
        Señales de fallo (toma lo que haya disponible en tus CSVs).
        Acepta columnas como: used_fact, retrieval_hit@k, recall@k, mrr@k, ndcg@k, ok, etc.
        """
        for key in row.keys():
            lk = key.lower()
            val = row.get(key, "").strip().lower()

            # Señales típicas
            if lk in ("used_fact", "retrieval_hit@k", "retrieval_hit", "hit@k"):
                if val in ("0", "false", "no", ""):
                    return True
            if lk.startswith("recall@") or lk.startswith("mrr@") or lk.startswith("ndcg@"):
                try:
                    f = float(val)
                    if f == 0.0:
                        return True
                except:
                    pass
            if lk in ("ok", "success"):
                if val in ("0", "false", "no"):
                    return True

        return False

    def _guess_query(self, row: Dict[str, str]) -> Optional[str]:
        """
        Intentamos obtener una 'query' o algo parecido del CSV.
        """
        for c in ["query", "q", "prompt", "input", "text", "goal"]:
            if c in row and row[c].strip():
                return row[c].strip()
        return None

    def build_from_glob(self, pattern: str, limit_per_file: int = 50) -> List[MicroTask]:
        """
        Lee CSVs que matchean el patrón y agrega micro-tareas para filas fallidas.
        """
        self.tasks.clear()
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    added = 0
                    for row in reader:
                        if not self._looks_like_failure(row):
                            continue
                        q = self._guess_query(row)
                        if not q:
                            continue

                        # Generamos tarea para memoria (por defecto)
                        self.tasks.append(
                            MicroTask(
                                goal=q,
                                hint="Recupera desde tu memoria; si no, intenta web.",
                                kind="memory",
                            )
                        )
                        added += 1
                        if added >= limit_per_file:
                            break
            except Exception:
                # Ignorar archivos que no sean CSV o estén corruptos
                continue
        return self.tasks

    def preview(self, n: int = 5) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tasks[:n]]
