"""
Stage 0: Repo Summarization
Takes a RepoSnapshot and asks an LLM to produce a structured project spec:
what it does, its architecture, key modules, and expected correctness
patterns. This spec is what Stage 1 (PR review) will check diffs against.

Provider-agnostic by design: swap `call_llm` to point at Anthropic, OpenAI,
or Gemini/Antigravity's API without touching anything else in this file.
"""

import os
import json
from dataclasses import dataclass, asdict
from src.ingest import RepoSnapshot

SYSTEM_PROMPT = """You are a senior software architect writing an onboarding \
spec for a new engineer joining this codebase. Given a file tree and the \
contents of key project files, produce a structured JSON summary.

Return ONLY valid JSON, no markdown fences, no commentary, matching this shape:

{
  "project_type": "short description, e.g. 'Full-stack web app, Python/FastAPI backend + React frontend'",
  "purpose": "1-3 sentences on what the project does",
  "architecture": "short paragraph on how the pieces fit together",
  "key_modules": [
    {"path": "path/or/dir", "role": "what this module is responsible for"}
  ],
  "conventions": [
    "notable patterns the codebase follows, e.g. 'all API routes validate input with Pydantic models'"
  ],
  "risk_areas": [
    "parts of the codebase where bugs would be especially costly, e.g. 'auth middleware', 'payment handling'"
  ]
}

If a field can't be determined from the given context, use your best \
inference from the file tree and naming conventions, and say so briefly \
rather than leaving it empty."""


@dataclass
class ProjectSpec:
    project_type: str
    purpose: str
    architecture: str
    key_modules: list
    conventions: list
    risk_areas: list
    repo_url: str
    commit_hash: str

    def to_dict(self):
        return asdict(self)


def build_user_prompt(snapshot: RepoSnapshot) -> str:
    signal_block = "\n\n".join(
        f"--- {fname} ---\n{content}"
        for fname, content in snapshot.signal_file_contents.items()
    )
    lang_block = ", ".join(f"{ext} ({count})" for ext, count in snapshot.languages.items())

    return f"""Repo: {snapshot.repo_url}
Total tracked files: {snapshot.total_files}
Language breakdown: {lang_block}

FILE TREE:
{snapshot.file_tree}

KEY FILE CONTENTS:
{signal_block}
"""


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Default implementation calls the Gemini API free tier (AI Studio).
    Get a free key at aistudio.google.com -- separate from Gemini app premium.

    To use Anthropic instead, swap this body for the commented block below.
    Everything else in this module stays the same either way.
    """
    import google.generativeai as genai  # pip install google-generativeai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-flash-lite-latest",
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_prompt)
    return response.text

    # --- Anthropic alternative (paid) ---
    # import anthropic  # pip install anthropic
    # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=2000,
    #     system=system_prompt,
    #     messages=[{"role": "user", "content": user_prompt}],
    # )
    # return response.content[0].text


def summarize_repo(snapshot: RepoSnapshot) -> ProjectSpec:
    """Main entry point: RepoSnapshot -> ProjectSpec via LLM call."""
    user_prompt = build_user_prompt(snapshot)
    raw = call_llm(SYSTEM_PROMPT, user_prompt)

    # Models sometimes wrap JSON in markdown fences despite instructions -- strip defensively.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)

    return ProjectSpec(
        project_type=data.get("project_type", ""),
        purpose=data.get("purpose", ""),
        architecture=data.get("architecture", ""),
        key_modules=data.get("key_modules", []),
        conventions=data.get("conventions", []),
        risk_areas=data.get("risk_areas", []),
        repo_url=snapshot.repo_url,
        commit_hash=snapshot.commit_hash,
    )
