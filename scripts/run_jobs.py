# scripts/run_jobs.py
from pathlib import Path
import json, time, sys, os, argparse
from executive.jobs import run_due_jobs, run_job_by_id
from root.settings import DATA_DIR

LOG = DATA_DIR / "jobs" / "run_log.jsonl"
LOCK = DATA_DIR / "jobs" / "runner.lock"

def _append(entry: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _rotate(max_bytes: int = 5_000_000):
    try:
        if LOG.exists() and LOG.stat().st_size > max_bytes:
            ts = int(time.time())
            dst = LOG.with_name(f"run_log.{ts}.jsonl")
            if dst.exists():
                dst.unlink()
            LOG.rename(dst)
    except Exception:
        # no detener corridas por rotación
        pass

class RunnerLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # lock por archivo (simple, portabilidad Windows)
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(self.fd, str(int(time.time())).encode("utf-8"))
            return True
        except FileExistsError:
            return False
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass

def _print_min(results):
    print(json.dumps(
        [{"job_id": r.get("job_id"), "ok": r.get("ok"), "latency_ms": r.get("latency_ms")} for r in results],
        ensure_ascii=False
    ))

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", metavar="JOB_ID", help="Forzar ejecución de un job específico")
    parser.add_argument("--help", action="store_true")
    args = parser.parse_args(argv)

    if args.help:
        print("Usage: python -m scripts.run_jobs [--run-id JOB_ID]")
        return 0

    now = int(time.time())
    _rotate()

    with RunnerLock(LOCK) as acquired:
        if not acquired and not args.run_id:
            # otra instancia corriendo: salida vacía para scheduler
            _print_min([])
            return 0

        results = []
        if args.run_id:
            r = run_job_by_id(args.run_id)
            results = [r]
            _append({"ts": now, **r})
        else:
            for r in run_due_jobs(now):
                _append({"ts": now, **r})
                results.append(r)

        _print_min(results)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
