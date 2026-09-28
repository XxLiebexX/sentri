"""
Stage 0: Spec Cache
Regenerating a full repo summary on every PR is wasteful and slow.
We cache the ProjectSpec by (repo_url, commit_hash) and only regenerate
when the repo's default branch has moved.

Swap this for Redis/Postgres later -- interface stays the same.
"""

import json
import hashlib
from pathlib import Path
from src.summarize import ProjectSpec

CACHE_DIR = Path(".stage0_cache")


def _cache_key(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode()).hexdigest()[:16]


def get_cached_spec(repo_url: str, commit_hash: str) -> ProjectSpec | None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(repo_url)}.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    if data.get("commit_hash") != commit_hash:
        # Repo has moved on -- cache is stale.
        return None

    return ProjectSpec(**{k: v for k, v in data.items() if k != "cached_at"})


def save_spec(spec: ProjectSpec) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(spec.repo_url)}.json"
    path.write_text(json.dumps(spec.to_dict(), indent=2))
