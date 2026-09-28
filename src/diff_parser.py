"""
Stage 1: Diff Parser
Turns a raw git diff into CodeChunk objects -- the changed lines plus
enough surrounding context (the full enclosing function) for an LLM to
actually judge correctness, not just pattern-match on a fragment.
"""

import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    file: str
    line_start: int
    line_end: int
    diff_text: str
    full_context: str
    spec_context: str = ""  # filled in later from Stage 0's ProjectSpec


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def get_git_diff(repo_path: str = ".", against: str = "HEAD~1") -> str:
    """Fetch a raw diff. Defaults to comparing against the previous commit --
    swap `against` for a branch name / PR base ref when wiring into GitHub."""
    result = subprocess.run(
        ["git", "-C", repo_path, "diff", against, "--unified=3"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return result.stdout


def parse_diff(diff_text: str) -> list[dict]:
    """
    Split a raw unified diff into per-file, per-hunk pieces.
    Returns a list of {file, line_start, line_end, diff_text} dicts --
    intentionally simple regex parsing, not a full diff AST, since we
    only need the line ranges and file paths to go find real context.
    """
    hunks = []
    current_file = None
    current_hunk_lines: list[str] = []
    current_start = current_end = None

    def flush():
        if current_file and current_hunk_lines:
            hunks.append({
                "file": current_file,
                "line_start": current_start,
                "line_end": current_end,
                "diff_text": "\n".join(current_hunk_lines),
            })

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            flush()
            current_file = line[6:].strip()
            current_hunk_lines = []
            continue

        if line.startswith("+++ /dev/null") or line.startswith("--- "):
            continue

        match = HUNK_HEADER_RE.match(line)
        if match:
            flush()
            current_start = int(match.group(1))
            length = int(match.group(2)) if match.group(2) else 1
            current_end = current_start + max(length - 1, 0)
            current_hunk_lines = [line]
            continue

        if current_hunk_lines is not None and (current_file):
            current_hunk_lines.append(line)

    flush()
    return hunks


def extract_function_context(file_path: str, line_start: int, line_end: int, repo_path: str = ".") -> str:
    """
    Grab the full function/block a changed hunk lives inside, not just the
    changed lines. Uses simple indentation-based scanning -- good enough
    for Python; for brace languages it falls back to a padded line window.
    Swap for tree-sitter later if you want it airtight across languages.
    """
    full_path = f"{repo_path.rstrip('/')}/{file_path}"
    try:
        with open(full_path, errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ""

    if not lines:
        return ""

    is_python = file_path.endswith(".py")

    if is_python:
        return _extract_python_block(lines, line_start, line_end)
    else:
        return _extract_padded_window(lines, line_start, line_end)


def _extract_python_block(lines: list[str], line_start: int, line_end: int) -> str:
    """Walk upward to the nearest `def`/`class` at or below the hunk's
    indentation, then walk downward until indentation returns to that level."""
    idx = max(line_start - 1, 0)
    idx = min(idx, len(lines) - 1)

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip())

    # Walk up to find the enclosing def/class
    start_idx = idx
    while start_idx > 0:
        stripped = lines[start_idx].lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
        start_idx -= 1
    block_indent = indent_of(lines[start_idx])

    # Walk down until indentation drops back to block level (end of function)
    end_idx = min(line_end, len(lines) - 1)
    while end_idx + 1 < len(lines):
        next_line = lines[end_idx + 1]
        if next_line.strip() and indent_of(next_line) <= block_indent:
            break
        end_idx += 1

    return "".join(lines[start_idx:end_idx + 1])


def _extract_padded_window(lines: list[str], line_start: int, line_end: int, pad: int = 15) -> str:
    """Fallback for non-Python files: just pad N lines above/below the hunk."""
    start = max(line_start - pad - 1, 0)
    end = min(line_end + pad, len(lines))
    return "".join(lines[start:end])


def build_chunks(diff_text: str, repo_path: str = ".") -> list[CodeChunk]:
    """Main entry point: raw diff -> list of CodeChunk with full context."""
    raw_hunks = parse_diff(diff_text)
    chunks = []

    for hunk in raw_hunks:
        context = extract_function_context(
            hunk["file"], hunk["line_start"], hunk["line_end"], repo_path
        )
        chunks.append(CodeChunk(
            file=hunk["file"],
            line_start=hunk["line_start"],
            line_end=hunk["line_end"],
            diff_text=hunk["diff_text"],
            full_context=context,
        ))

    return chunks


if __name__ == "__main__":
    # Quick manual test: run from inside a git repo with uncommitted changes
    # or at least one prior commit.
    diff = get_git_diff(".", against="HEAD~1")
    chunks = build_chunks(diff, repo_path=".")
    for c in chunks:
        print(f"\n=== {c.file} (lines {c.line_start}-{c.line_end}) ===")
        print(c.full_context[:500])
