# scripts/metrics_run_and_commit.py
from __future__ import annotations
import argparse, csv, glob, os, sys
from pathlib import Path
from datetime import datetime

# Depende solo de git_utils y del CSV generado por compute_and_log_memory_metrics
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add repo root
from tools.git_utils import is_git_repo, git_add, git_commit, git_status_porcelain  # type: ignore

def newest_metrics_csv(out_dir: Path) -> Path | None:
    files = sorted(out_dir.glob("memory_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def last_row(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8", newline="") as fh:
        r = list(csv.DictReader(fh))
        return r[-1] if r else None

def main():
    ap = argparse.ArgumentParser(description="Run metrics test and git-commit latest CSV.")
    ap.add_argument("--out-dir", default="logs/benchmarks", help="Carpeta donde se escriben los CSV")
    ap.add_argument("--k", type=int, default=2, help="k esperado solo para el mensaje")
    ap.add_argument("--run-test", action="store_true", help="Ejecuta el test antes de commitear")
    ap.add_argument("--stage-all", action="store_true", help="Hace 'git add -A' (además del CSV)")
    ap.add_argument("--extra", nargs="*", default=[], help="Rutas extra para git add")
    ap.add_argument("--allow-empty", action="store_true", help="Permite commit vacío")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    if not is_git_repo(str(repo_root)):
        print("⚠️  No estás dentro de un repositorio git. Aborto.")
        sys.exit(1)

    if args.run_test:
        # Ejecuta SOLO el test extendido de métricas; si no tienes pytest, omite --run-test
        import subprocess
        cmd = ["pytest", "-q", "tests/test_memory_metrics.py::test_compute_and_log_extended_metrics"]
        print(f"▶ Ejecutando: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print("❌ El test falló. No se hará commit.")
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = newest_metrics_csv(out_dir)
    if not csv_path:
        print(f"⚠️  No encontré CSVs en {out_dir}. Nada que commitear (a menos que uses --stage-all/--extra).")

    msg_suffix = ""
    if csv_path:
        row = last_row(csv_path) or {}
        rec = row.get("recall@k")
        mrr = row.get("mrr@k")
        ndcg = row.get("ndcg@k")
        used = row.get("used_fact")
        rhit = row.get("retrieval_hit@k")
        rid = row.get("run_id") or "n/a"
        msg_suffix = f"run_id={rid} recall@{args.k}={rec} mrr@{args.k}={mrr} ndcg@{args.k}={ndcg} used_fact={used} hit@{args.k}={rhit}"

    # Stage
    staged = []
    if csv_path:
        print(f"➕ git add {csv_path}")
        git_add([str(csv_path)])
        staged.append(str(csv_path))

    if args.extra:
        print(f"➕ git add {' '.join(args.extra)}")
        git_add(args.extra)
        staged.extend(args.extra)

    if args.stage_all:
        print("➕ git add -A")
        git_add(all=True)

    # Commit
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"metrics: {now}"
    if msg_suffix:
        msg += f" | {msg_suffix}"

    status = git_status_porcelain()
    will_commit = bool(status.strip()) or args.allow_empty
    if not will_commit:
        print("ℹ️  No hay cambios para commitear (usa --allow-empty si lo necesitas).")
        sys.exit(0)

    print(f"✅ git commit -m \"{msg}\"")
    out = git_commit(msg, allow_empty=args.allow_empty)
    print(out)

if __name__ == "__main__":
    main()
