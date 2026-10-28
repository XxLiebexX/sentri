# Sentri — Hybrid AI Code Review & Security Agent

Sentri is an intelligent, cost-effective code review agent designed to catch security vulnerabilities and architectural convention violations. It uses a **"cheap filter, expensive escalation"** architecture: running a fast, locally fine-tuned small language model (SLM) on your machine first, and escalating to Google Gemini Flash only when confidence is low or when vulnerabilities are detected.

---

## 🏗️ Architecture

```
                       ┌─────────────────────────┐
                       │   Pull Request / Code   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Static Analysis (Semgrep)│
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Local Model (Qwen 1.5B) │
                       │    LoRA Fine-Tuned      │
                       └────────────┬────────────┘
                                    │
                  Confidence >= 0.98 & Verdict == "safe"?
                                   / \
                            YES  /     \  NO (Low confidence / Vulnerable)
                                /       \
                               ▼         ▼
                      ┌─────────────┐   ┌──────────────────────┐
                      │ Done (Free) │   │ Escalate to Gemini   │
                      └─────────────┘   └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │ Final Review Verdict │
                                        └──────────────────────┘
```

---

## 🧩 Key Stages & Components

1. **Stage 0: Ground-Truth Spec Generator (`src/ingest.py`, `src/summarize.py`, `main.py`)**:
   - Clones a repo, parses key signal files (`README`, `requirements.txt`, entrypoints, Dockerfiles, etc.), and generates a structured JSON architecture specification.
   - Caches specs by commit hash so unchanged repos are never re-analyzed.
2. **Local Model Engine (`sentri-local-model/`, `src/local_model.py`)**:
   - Fine-tuned `Qwen2.5-Coder-1.5B-Instruct` using LoRA/PEFT on BigVul security datasets (`finetune/prepare_dataset.py`, `finetune/train_mac.py`).
   - Runs locally on Apple Silicon MPS or CPU with token-level confidence estimation.
3. **Stage 1 PR & Diff Reviewer (`src/diff_parser.py`, `src/llm_reviewer.py`, `src/static_analysis.py`)**:
   - Parses unified git diffs, extracts enclosing function context, and incorporates Semgrep static analysis findings into the review prompt.
4. **GitHub Bot (`src/github_bot.py`)**:
   - Webhook server listening for pull request events to post line-by-line review comments directly to GitHub.
5. **Full Repository Scanner (`src/scan_repo.py`)**:
   - Clones and audits an entire codebase file-by-file with static checks + hybrid AI review.
6. **Web Dashboard (`backend/main.py`, `frontend/`)**:
   - FastAPI server + modern web interface providing side-by-side local vs. Gemini comparisons and whole-repo scanning.
7. **Evaluation Benchmark (`eval/compare_strategies.py`)**:
   - Measures precision, recall, and API call savings across local-only, cloud-only, and hybrid routing strategies.

---

## 🚀 Quickstart

### 1. Setup

```bash
git clone https://github.com/XxLiebexX/sentri.git
cd sentri
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
```

### 2. Run the Web Dashboard

Start the backend API:
```bash
uvicorn backend.main:app --port 8000
```

Serve the frontend:
```bash
python3 -m http.server 3001 --directory frontend
```
Open **`http://localhost:3001`** in your browser.

---

## 💻 CLI Usage

- **Generate Stage 0 Architecture Spec**:
  ```bash
  python main.py https://github.com/pallets/flask
  ```

- **Scan a Full Repository**:
  ```bash
  python -m src.scan_repo https://github.com/pallets/flask --max-files 5
  ```

- **Run Routing Strategy Evaluation**:
  ```bash
  python eval/compare_strategies.py
  ```

- **Start GitHub PR Webhook Bot**:
  ```bash
  export GITHUB_TOKEN="ghp_xxx"
  export GITHUB_WEBHOOK_SECRET="secret"
  uvicorn src.github_bot:app --port 8001
  ```

---

## 📊 Evaluation & Efficiency

By filtering safe code locally with a high confidence threshold (`0.98`), Sentri eliminates up to **70-80% of routine cloud LLM API calls** while maintaining high vulnerability recall through Gemini escalation.

---

## 🔍 Known Limitations & Future Roadmap

- **Monorepos**: Signal-file-based ingestion can miss context in complex monorepo layouts. Deep AST code graphs (via Tree-sitter / LSP) will be integrated to map multi-package dependencies.
- **Private Repositories**: Currently assumes public repos or pre-configured local Git credentials. Adding GitHub App OAuth token generation is on the roadmap.
- **Automated Fix Generation**: Expanding from detection to generating one-click remediation patch suggestions in PR comments.
