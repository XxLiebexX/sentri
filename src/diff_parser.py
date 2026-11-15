"""
Stage 1: Diff Parser
Turns a raw git diff into CodeChunk objects -- the changed lines plus
enough surrounding context (the full enclosing function) for an LLM to
actually judge correctness, not just pattern-match on a fragment.
"""

import re
import subprocess
from pathlib import Path
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


BRACE_EXTENSIONS = {
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".swift", ".kt",
}

CONTROL_FLOW_RE = re.compile(r"^\s*(if|else|for|while|switch|case|try|catch|finally|do)\b")


def extract_function_context(file_path: str, line_start: int, line_end: int, repo_path: str = ".") -> str:
    """
    Grab the full function/block a changed hunk lives inside, not just the
    changed lines. Uses Tree-Sitter for Python, brace-matching for C-style languages
    (JS/TS/Java/Go/C++), and falls back to a padded line window for others.
    """
    full_path = f"{repo_path.rstrip('/')}/{file_path}"
    try:
        with open(full_path, errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ""

    if not lines:
        return ""

    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return _extract_python_block(lines, line_start, line_end)
    elif ext in BRACE_EXTENSIONS:
        brace_block = _extract_brace_block(lines, line_start, line_end)
        if brace_block:
            return brace_block
        return _extract_padded_window(lines, line_start, line_end)
    else:
        return _extract_padded_window(lines, line_start, line_end)


def _extract_brace_block(lines: list[str], line_start: int, line_end: int, max_scan: int = 150) -> str | None:
    """
    Extract the enclosing function or method block for brace-delimited languages
    (JavaScript, TypeScript, Go, Java, C/C++, Rust, etc.).
    Scans upward to locate the enclosing function header, then balances braces forward.
    """
    start_idx = max(0, line_start - 1)

    depth = 0
    enclosing_open_line = None

    # Scan backwards to locate the opening brace of the enclosing function
    for idx in range(start_idx, max(-1, start_idx - max_scan), -1):
        line = lines[idx]
        clean = line.split("//")[0]  # strip single-line comments
        for ch in reversed(clean):
            if ch == "}":
                depth += 1
            elif ch == "{":
                if depth > 0:
                    depth -= 1
                else:
                    enclosing_open_line = idx
                    # If this block is a control flow statement (e.g. if/for), continue climbing
                    if not CONTROL_FLOW_RE.match(clean.strip()):
                        break
        if enclosing_open_line is not None and not CONTROL_FLOW_RE.match(clean.strip()):
            break

    if enclosing_open_line is None:
        return None

    # Walk backward to capture multiline signatures (parameters, annotations/decorators)
    sig_start = enclosing_open_line
    while sig_start > 0:
        prev = lines[sig_start - 1].strip()
        if not prev or prev.endswith(";") or prev.endswith("{") or prev.endswith("}"):
            break
        sig_start -= 1

    # Check for preceding annotations/decorators (e.g., @Override, @Transactional)
    while sig_start > 0 and lines[sig_start - 1].strip().startswith("@"):
        sig_start -= 1

    # Scan forward to locate the matching closing brace
    f_depth = 0
    end_line = None
    for idx in range(enclosing_open_line, len(lines)):
        clean = lines[idx].split("//")[0]
        for ch in clean:
            if ch == "{":
                f_depth += 1
            elif ch == "}":
                f_depth -= 1
                if f_depth == 0:
                    end_line = idx
                    break
        if end_line is not None:
            break

    if end_line is None:
        end_line = min(len(lines) - 1, line_end + 15)

    return "".join(lines[sig_start:end_line + 1])


def _extract_python_block(lines: list[str], line_start: int, line_end: int) -> str:
    """Use tree-sitter to find the exact enclosing function or class for the changed hunk."""
    try:
        import tree_sitter_python
        from tree_sitter import Language, Parser
    except ImportError:
        # Fallback if not installed
        return _extract_padded_window(lines, line_start, line_end)

    source_code = "".join(lines).encode("utf8")
    
    PY_LANGUAGE = Language(tree_sitter_python.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source_code)
    
    # line_start and line_end are 1-indexed, tree-sitter points are 0-indexed
    start_point = (max(0, line_start - 1), 0)
    end_point = (min(len(lines) - 1, line_end - 1), len(lines[min(len(lines) - 1, line_end - 1)]))

    # Walk the tree to find the smallest enclosing function_definition or class_definition
    node = tree.root_node.descendant_for_point_range(start_point, end_point)
    while node:
        if node.type in ("function_definition", "class_definition"):
            break
        node = node.parent
        
    if node:
        # node.start_point[0] is the start line index
        return "".join(lines[node.start_point[0]:node.end_point[0] + 1])
    else:
        # If no enclosing block found, fallback to padded window
        return _extract_padded_window(lines, line_start, line_end)


def _extract_padded_window(lines: list[str], line_start: int, line_end: int, pad: int = 15) -> str:
    """Fallback for non-Python files: just pad N lines above/below the hunk."""
    start = max(line_start - pad - 1, 0)
    end = min(line_end + pad, len(lines))
    return "".join(lines[start:end])


def slice_spec_for_file(spec, file_path: str) -> str:
    """
    Extract the relevant slice of a Stage 0 ProjectSpec for a specific file being reviewed.
    Matches key modules, relevant conventions, and risk areas to give the LLM
    ground-truth context about this specific file within the overall architecture.
    """
    if not spec:
        return ""

    if hasattr(spec, "to_dict"):
        spec_dict = spec.to_dict()
    elif isinstance(spec, dict):
        spec_dict = spec
    else:
        return ""

    sections = []

    project_type = spec_dict.get("project_type", "").strip()
    purpose = spec_dict.get("purpose", "").strip()
    if project_type or purpose:
        sections.append(f"PROJECT OVERVIEW:\nType: {project_type}\nPurpose: {purpose}")

    # Check key modules for path match
    matched_modules = []
    norm_file = file_path.strip().lstrip("/")
    for mod in spec_dict.get("key_modules", []):
        if isinstance(mod, dict):
            mod_path = mod.get("path", "").strip().lstrip("/")
            if mod_path and (mod_path in norm_file or norm_file.startswith(mod_path)):
                role = mod.get("role", "")
                matched_modules.append(f"- {mod_path}: {role}")
    if matched_modules:
        sections.append("MATCHED MODULE ROLE:\n" + "\n".join(matched_modules))

    # Relevant conventions
    conventions = spec_dict.get("conventions", [])
    if conventions:
        sections.append("PROJECT CONVENTIONS:\n" + "\n".join(f"- {c}" for c in conventions))

    # Risk areas
    risk_areas = spec_dict.get("risk_areas", [])
    if risk_areas:
        sections.append("HIGH-RISK AREAS TO WATCH:\n" + "\n".join(f"- {r}" for r in risk_areas))

    return "\n\n".join(sections).strip()


def build_chunks(diff_text: str, repo_path: str = ".", spec=None) -> list[CodeChunk]:
    """Main entry point: raw diff -> list of CodeChunk with full context and sliced spec context."""
    raw_hunks = parse_diff(diff_text)
    chunks = []

    for hunk in raw_hunks:
        context = extract_function_context(
            hunk["file"], hunk["line_start"], hunk["line_end"], repo_path
        )
        spec_context = slice_spec_for_file(spec, hunk["file"]) if spec else ""
        chunks.append(CodeChunk(
            file=hunk["file"],
            line_start=hunk["line_start"],
            line_end=hunk["line_end"],
            diff_text=hunk["diff_text"],
            full_context=context,
            spec_context=spec_context,
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
