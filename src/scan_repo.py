"""
Full Repository Scan
Point this at a repo URL and it:
  1. Clones the repo and gets Stage 0's architecture overview
  2. Walks every source file (not just a diff)
  3. Runs static analysis (semgrep) across all of them
  4. Runs the local model + Gemini router on each file for deeper review
  5. Produces one combined report: project overview + every flaw found,
     with file/line references and which reviewer (local/gemini/static)
     caught each one

This is different from Stage 1 (llm_reviewer.py), which only reviews a
PR\'s diff. This scans the whole codebase as it stands right now.

Run: python3 -m src.scan_repo <repo_url> [--max-files N] [--skip-ai]
"""

import os
import sys
import json
import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest import IGNORE_DIRS, clone_repo, get_commit_hash
from src.summarize import summarize_repo, ProjectSpec
from src.ingest import ingest_repo
from src.static_analysis import run_semgrep, StaticFinding

SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rs", ".swift", ".kt",
}

MAX_FILE_SIZE_BYTES = 200_000


@dataclass
class RepoFinding:
    file: str
    line: int
    severity: str
    category: str
    message: str
    source: str


@dataclass
class RepoScanReport:
    repo_url: str
    commit_hash: str
    project_spec: dict
    files_scanned: int
    files_skipped: int
    findings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "repo_url": self.repo_url,
            "commit_hash": self.commit_hash,
            "project_spec": self.project_spec,
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "total_findings": len(self.findings),
            "findings": [f.__dict__ for f in self.findings],
        }


def list_source_files(local_path: str) -> list:
    source_files = []
    root = Path(local_path)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            ext = Path(fname).suffix
            if ext not in SOURCE_EXTENSIONS:
                continue
            full_path = Path(dirpath) / fname
            try:
                if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            rel_path = str(full_path.relative_to(root))
            source_files.append(rel_path)

    return sorted(source_files)


def run_static_analysis_on_repo(local_path: str, files: list) -> list:
    if not files:
        return []
    print(f"[scan] Running static analysis on {len(files)} files...")
    return run_semgrep(files, repo_path=local_path)


async def run_ai_review_on_file(file_path: str, content: str, local_reviewer, spec_context: str) -> list:
    from src.router import review_code

    prompt = f"""Review this file for bugs, security vulnerabilities, and issues.

PROJECT CONTEXT:
{spec_context}

FILE: {file_path}

CODE:
{content[:6000]}
"""

    result = await review_code(prompt, local_reviewer)

    if result.verdict in ("vulnerable", "unknown"):
        return [RepoFinding(
            file=file_path,
            line=1,
            severity="error" if result.verdict == "vulnerable" else "info",
            category=result.cwe,
            message=result.explanation,
            source=result.source,
        )]
    return []


async def scan_repo(repo_url: str, max_files: int = 20, skip_ai: bool = False, skip_spec: bool = False, local_reviewer=None) -> RepoScanReport:
    print(f"[scan] Cloning {repo_url}...")
    local_path = clone_repo(repo_url)

    try:
        commit_hash = get_commit_hash(local_path)

        project_spec_dict = {}
        spec_context = "No project spec available."
        if not skip_spec:
            print("[scan] Generating project overview (Stage 0)...")
            snapshot = ingest_repo(repo_url)
            spec = summarize_repo(snapshot)
            project_spec_dict = spec.to_dict()
            spec_context = f"{spec.project_type}: {spec.purpose}"

        all_files = list_source_files(local_path)
        files_to_scan = all_files[:max_files]
        files_skipped = len(all_files) - len(files_to_scan)

        if files_skipped > 0:
            print(f"[scan] Found {len(all_files)} source files, scanning first {max_files} (skipping {files_skipped})")
        else:
            print(f"[scan] Found {len(all_files)} source files, scanning all of them")

        static_findings = run_static_analysis_on_repo(local_path, files_to_scan)

        findings = [
            RepoFinding(file=f.file, line=f.line, severity=f.severity.lower(), category=f.rule_id, message=f.message, source="static")
            for f in static_findings
        ]

        if not skip_ai:
            if local_reviewer is None:
                from src.local_model import LocalReviewer
                import asyncio
                print("[scan] Loading local model for AI review...")
                local_reviewer = await asyncio.to_thread(LocalReviewer)

            for i, rel_path in enumerate(files_to_scan, 1):
                full_path = Path(local_path) / rel_path
                try:
                    content = full_path.read_text(errors="ignore")
                except Exception:
                    continue
                if not content.strip():
                    continue

                print(f"[scan] Reviewing file ({i}/{len(files_to_scan)}): {rel_path}")
                try:
                    res = await run_ai_review_on_file(rel_path, content, local_reviewer, spec_context)
                    findings.extend(res)
                except Exception as e:
                    print(f"[scan] Error reviewing {rel_path}: {e}")

        return RepoScanReport(
            repo_url=repo_url, commit_hash=commit_hash, project_spec=project_spec_dict,
            files_scanned=len(files_to_scan), files_skipped=files_skipped, findings=findings,
        )

    finally:
        subprocess.run(["rm", "-rf", local_path], capture_output=True)


def print_report(report: RepoScanReport):
    print("\n" + "=" * 70)
    print(f"SCAN REPORT: {report.repo_url}")
    print("=" * 70)

    if report.project_spec:
        print(f"\nProject type: {report.project_spec.get('project_type', 'unknown')}")
        print(f"Purpose: {report.project_spec.get('purpose', 'unknown')}")

    print(f"\nFiles scanned: {report.files_scanned} (skipped: {report.files_skipped})")
    print(f"Total findings: {len(report.findings)}")

    if not report.findings:
        print("\nNo issues found in the scanned files.")
        return

    by_severity = {}
    for f in report.findings:
        by_severity.setdefault(f.severity, []).append(f)

    for severity in ["error", "warning", "info"]:
        items = by_severity.get(severity, [])
        if not items:
            continue
        print(f"\n--- {severity.upper()} ({len(items)}) ---")
        for f in items:
            print(f"  {f.file}:{f.line} [{f.category}] ({f.source})")
            print(f"    {f.message[:200]}")


def main():
    import asyncio
    parser = argparse.ArgumentParser(description="Full repository vulnerability + issue scan")
    parser.add_argument("repo_url", help="Git URL to scan")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--skip-spec", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    report = asyncio.run(scan_repo(args.repo_url, max_files=args.max_files, skip_ai=args.skip_ai, skip_spec=args.skip_spec))
    print_report(report)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report written to {args.json_out}")


if __name__ == "__main__":
    main()
