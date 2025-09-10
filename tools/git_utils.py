# tools/git_utils.py
from __future__ import annotations
import subprocess
from typing import List, Optional

def _run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
    out = subprocess.run(
        cmd, cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return out.stdout

def is_git_repo(path: str = ".") -> bool:
    try:
        return _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path).strip() == "true"
    except Exception:
        return False

def git_status_porcelain(cwd: str = ".") -> str:
    return _run(["git", "status", "--porcelain"], cwd=cwd, check=False)

def git_add(paths: List[str] | None = None, all: bool = False, cwd: str = ".") -> str:
    if all:
        return _run(["git", "add", "-A"], cwd=cwd)
    if not paths:
        return ""
    return _run(["git", "add", *paths], cwd=cwd)

def git_commit(message: str, allow_empty: bool = False, cwd: str = ".") -> str:
    cmd = ["git", "commit", "-m", message]
    if allow_empty:
        cmd.append("--allow-empty")
    return _run(cmd, cwd=cwd, check=False)  # si no hay cambios, no truena el proceso
