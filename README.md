# Stage 0 — Repo Overview

This is the first stage of the AI code-review agent. Give it a git URL,
and it clones the repo, walks its structure, and asks an LLM to produce
a structured spec describing what the project is and how it's built.

That spec becomes the "ground truth" Stage 1 (PR review) checks diffs
against — so the review agent understands *this specific project's*
architecture and conventions, not just generic code patterns.

## What it does

1. **`ingest.py`** — clones the repo, builds a depth-limited file tree,
   pulls the contents of key signal files (README, package.json,
   requirements.txt, entry points, Dockerfile, etc.), and does a rough
   language breakdown by file extension.
2. **`summarize.py`** — sends that ingested context to an LLM with a
   prompt asking for a structured JSON spec: project type, purpose,
   architecture, key modules, coding conventions, and risk areas.
3. **`cache.py`** — caches the spec by repo URL + commit hash, so you're
   not re-summarizing an unchanged repo on every run (this matters a lot
   once Stage 1 calls this on every PR).
4. **`main.py`** — CLI glue: `python main.py <git_url>`.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run it

```bash
python main.py https://github.com/pallets/flask
```

Output is a JSON spec like:

```json
{
  "project_type": "Python web framework",
  "purpose": "A lightweight WSGI web application framework...",
  "architecture": "Core Flask app object wraps Werkzeug routing...",
  "key_modules": [
    {"path": "src/flask/app.py", "role": "Main Flask application class and request handling"}
  ],
  "conventions": ["Extensive use of decorators for route registration", "..."],
  "risk_areas": ["Session/cookie signing logic", "..."]
}
```

## Swapping the LLM provider

Everything routes through `call_llm()` in `summarize.py`. It currently
calls the Anthropic API. To use Gemini (e.g. if driving this from
Google Antigravity) or OpenAI instead, replace the body of that one
function — the prompt-building, caching, and CLI layers don't need to
change.

## Next: Stage 1

Stage 1 (PR review) will:
- Fetch the relevant slice of this spec for the modules a PR actually touches
- Feed it alongside the diff + static analysis findings to the review LLM
- Flag not just generic bugs, but violations of *this project's* own
  patterns (e.g. "this endpoint skips the auth check every other route uses")

## Known limitations (be upfront about these in interviews — it shows judgment)

- Signal-file-based ingestion works well for typical web/app repos but
  will miss context in unusually-structured monorepos — worth testing
  against a few different repo shapes.
- No handling yet for private repos requiring auth — `clone_repo` assumes
  a public URL or that git credentials are already configured locally.
- Spec quality is only as good as the README/config files present; very
  sparse repos will get a thinner, more inferred spec.
