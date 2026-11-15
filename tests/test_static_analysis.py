import json
import unittest
from unittest.mock import patch, MagicMock

from src.static_analysis import (
    StaticFinding,
    findings_for_chunk,
    format_findings_for_prompt,
    run_semgrep,
)
from src.diff_parser import CodeChunk


class TestStaticAnalysis(unittest.TestCase):
    def test_findings_for_chunk_filtering(self):
        chunk = CodeChunk(
            file="app/routes.py",
            line_start=20,
            line_end=40,
            diff_text="",
            full_context="",
        )

        all_findings = [
            # In range
            StaticFinding(file="app/routes.py", line=25, severity="ERROR", rule_id="sql-inject", message="SQLi"),
            # Out of range (before)
            StaticFinding(file="app/routes.py", line=10, severity="WARNING", rule_id="unused-import", message="Unused"),
            # Out of range (after)
            StaticFinding(file="app/routes.py", line=55, severity="INFO", rule_id="todo", message="Todo"),
            # Different file
            StaticFinding(file="app/models.py", line=30, severity="ERROR", rule_id="hardcoded-key", message="Key"),
        ]

        filtered = findings_for_chunk(chunk, all_findings)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].rule_id, "sql-inject")
        self.assertEqual(filtered[0].line, 25)

    def test_format_findings_for_prompt(self):
        # Empty findings
        empty_text = format_findings_for_prompt([])
        self.assertEqual(empty_text, "No static analysis findings for this section.")

        # Non-empty findings
        findings = [
            StaticFinding(file="a.py", line=12, severity="ERROR", rule_id="cwe-78", message="Command injection"),
            StaticFinding(file="a.py", line=15, severity="WARNING", rule_id="cwe-79", message="XSS hazard"),
        ]
        text = format_findings_for_prompt(findings)
        self.assertIn("- [ERROR] cwe-78: Command injection (line 12)", text)
        self.assertIn("- [WARNING] cwe-79: XSS hazard (line 15)", text)

    def test_run_semgrep_empty_paths(self):
        self.assertEqual(run_semgrep([]), [])

    @patch("subprocess.run")
    def test_run_semgrep_parses_json(self, mock_run):
        mock_output = {
            "results": [
                {
                    "path": "/repo/src/main.py",
                    "start": {"line": 42},
                    "check_id": "security.bandit.B608",
                    "extra": {
                        "severity": "ERROR",
                        "message": "Possible SQL injection vector",
                    },
                }
            ]
        }
        mock_proc = MagicMock()
        mock_proc.returncode = 1  # Semgrep exits with 1 when findings are present
        mock_proc.stdout = json.dumps(mock_output)
        mock_run.return_value = mock_proc

        findings = run_semgrep(["src/main.py"], repo_path="/repo")
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.file, "src/main.py")
        self.assertEqual(finding.line, 42)
        self.assertEqual(finding.severity, "ERROR")
        self.assertEqual(finding.rule_id, "security.bandit.B608")


if __name__ == "__main__":
    unittest.main()
