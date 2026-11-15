import unittest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from backend.main import app
import backend.main as backend_module
from src.local_model import LocalVerdict
from src.router import ReviewResult


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("model_loaded", data)

    def test_review_empty_code_returns_400(self):
        resp = self.client.post("/api/review", json={"code": "   "})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No code provided", resp.json().get("detail", ""))

    def test_scan_repo_empty_url_returns_400(self):
        resp = self.client.post("/api/scan-repo", json={"repo_url": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No repo URL provided", resp.json().get("detail", ""))

    def test_overview_empty_url_returns_400(self):
        resp = self.client.post("/api/overview", json={"repo_url": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No repo URL provided", resp.json().get("detail", ""))

    @patch("backend.main.escalate_to_gemini", new_callable=AsyncMock)
    def test_review_endpoint_success_flow(self, mock_gemini):
        mock_gemini.return_value = ReviewResult(
            verdict="vulnerable",
            cwe="CWE-89",
            explanation="SQL Injection vulnerability",
            source="gemini",
            suggested_fix="cursor.execute('...', (x,))",
        )

        # Mock a loaded local reviewer
        mock_local = unittest.mock.MagicMock()
        mock_local.review.return_value = LocalVerdict(
            verdict="safe",
            cwe="none",
            explanation="Looked ok to local",
            confidence=0.92,  # below 0.98 -> triggers escalation
            suggested_fix=None,
        )

        with patch.object(backend_module, "_local_reviewer", mock_local):
            resp = self.client.post(
                "/api/review",
                json={"code": "query = 'SELECT * FROM users WHERE id=' + uid"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["escalated"])
            self.assertEqual(data["final_source"], "gemini")
            self.assertEqual(data["gemini"]["verdict"], "vulnerable")
            self.assertEqual(data["gemini"]["cwe"], "CWE-89")
            self.assertEqual(data["gemini"]["suggested_fix"], "cursor.execute('...', (x,))")


if __name__ == "__main__":
    unittest.main()
