"""
Stage 0: Repo Ingestion
Clones a git repo, walks its structure, and extracts the signals an LLM
needs to understand what the project does and how it's organized.
"""

import os
import subprocess
import tempfile
import fnmatch
from pathlib import Path
from dataclasses import dataclass, field

# Directories we never want to walk into (noise, not signal)
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".next", "target", "vendor", ".idea", ".vscode",
    "coverage", ".pytest_cache", ".mypy_cache", "egg-info",
}

# Files that strongly hint at project type / entry points
SIGNAL_FILES = [
    "README.md", "README.rst", "README.txt", "readme.md",
    "package.json", "requirements.txt", "pyproject.toml", "Pipfile",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "main.py", "app.py", "index.js", "index.ts", "server.js",
    "manage.py", "wsgi.py", "asgi.py",
    ".env.example", "Makefile",
]

# Cap how much raw content we send to the LLM per file, and in total.
MAX_FILE_CHARS = 4000
MAX_TOTAL_CHARS = 40000


@dataclass
class RepoSnapshot:
    """Everything Stage 0 gathers about a repo before summarizing."""
    repo_url: str
    commit_hash: str
    file_tree: str
    signal_file_contents: dict = field(default_factory=dict)
    languages: dict = field(default_factory=dict)  # extension -> file count
    total_files: int = 0


def clone_repo(repo_url: str, dest_dir: str | None = None) -> str:
    """Shallow-clone a repo (depth=1, fast) into a temp dir. Returns local path."""
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="stage0_")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest_dir],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return dest_dir


def get_commit_hash(local_path: str) -> str:
    """Used as a cache key so we don't re-summarize an unchanged repo."""
    result = subprocess.run(
        ["git", "-C", local_path, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_file_tree(local_path: str, max_depth: int = 4) -> str:
    """
    Produce a compact, indented text tree of the repo.
    Depth-limited and ignore-filtered so it stays small for large repos.
    """
    lines = []
    root = Path(local_path)

    def walk(dir_path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if entry.name in IGNORE_DIRS or entry.name.startswith("."):
                if entry.name not in (".env.example",):
                    continue
            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{entry.name}")

    walk(root, 0)
    return "\n".join(lines)


def collect_signal_files(local_path: str) -> dict:
    """Read the small set of files that best explain what the project is."""
    contents = {}
    total_chars = 0

    for fname in SIGNAL_FILES:
        fpath = Path(local_path) / fname
        if fpath.exists() and fpath.is_file():
            try:
                text = fpath.read_text(errors="ignore")
            except Exception:
                continue
            text = text[:MAX_FILE_CHARS]
            if total_chars + len(text) > MAX_TOTAL_CHARS:
                break
            contents[fname] = text
            total_chars += len(text)

    return contents


def count_languages(local_path: str) -> dict:
    """Rough language breakdown by file extension, for quick project-type signal."""
    counts: dict = {}
    root = Path(local_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            ext = Path(fname).suffix
            if ext:
                counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15])


def ingest_repo(repo_url: str) -> RepoSnapshot:
    """Main entry point for Stage 0 ingestion: clone -> walk -> extract signals."""
    local_path = clone_repo(repo_url)
    try:
        commit_hash = get_commit_hash(local_path)
        file_tree = build_file_tree(local_path)
        signal_files = collect_signal_files(local_path)
        languages = count_languages(local_path)
        total_files = sum(languages.values())

        return RepoSnapshot(
            repo_url=repo_url,
            commit_hash=commit_hash,
            file_tree=file_tree,
            signal_file_contents=signal_files,
            languages=languages,
            total_files=total_files,
        )
    finally:
        # Clean up the clone; we only needed it transiently.
        subprocess.run(["rm", "-rf", local_path], capture_output=True)
