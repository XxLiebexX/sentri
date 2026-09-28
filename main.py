"""
Stage 0 CLI: give it a git URL, get back a project spec.

Usage:
    python main.py https://github.com/some/repo
    python main.py https://github.com/some/repo --no-cache
"""

import sys
import json
import argparse

from src.ingest import ingest_repo, get_commit_hash, clone_repo
from src.summarize import summarize_repo
from src.cache import get_cached_spec, save_spec


def run(repo_url: str, use_cache: bool = True) -> dict:
    print(f"[stage0] Cloning {repo_url} ...")
    snapshot = ingest_repo(repo_url)
    print(f"[stage0] Commit {snapshot.commit_hash[:8]} | "
          f"{snapshot.total_files} files | "
          f"{len(snapshot.signal_file_contents)} signal files found")

    if use_cache:
        cached = get_cached_spec(repo_url, snapshot.commit_hash)
        if cached:
            print("[stage0] Using cached spec (repo unchanged since last run)")
            return cached.to_dict()

    print("[stage0] Generating project spec via LLM ...")
    spec = summarize_repo(snapshot)

    if use_cache:
        save_spec(spec)

    return spec.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Stage 0: repo -> project spec")
    parser.add_argument("repo_url", help="Git URL to analyze")
    parser.add_argument("--no-cache", action="store_true", help="Force regeneration")
    args = parser.parse_args()

    result = run(args.repo_url, use_cache=not args.no_cache)
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
