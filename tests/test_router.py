import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from src.router import review_code, ReviewResult, CONFIDENCE_THRESHOLD
from src.local_model import LocalVerdict


class MockLocalReviewer:
    def __init__(self, verdict: str, confidence: float, cwe: str = "none", explanation: str = "ok", suggested_fix: str | None = None):
        self._verdict = verdict
        self._confidence = confidence
        self._cwe = cwe
        self._explanation = explanation
        self._suggested_fix = suggested_fix

    def review(self, code: str) -> LocalVerdict:
        return LocalVerdict(
            verdict=self._verdict,
            confidence=self._confidence,
            cwe=self._cwe,
            explanation=self._explanation,
            suggested_fix=self._suggested_fix,
        )


class TestRouter(unittest.IsolatedAsyncioTestCase):
    @patch("src.router.escalate_to_gemini", new_callable=AsyncMock)
    async def test_confident_safe_stays_local(self, mock_escalate):
        # Local model is confident (0.99 >= 0.98) and safe
        reviewer = MockLocalReviewer(verdict="safe", confidence=0.99)
        result = await review_code("def foo(): pass", reviewer)

        self.assertEqual(result.source, "local")
        self.assertEqual(result.verdict, "safe")
        mock_escalate.assert_not_called()

    @patch("src.router.escalate_to_gemini", new_callable=AsyncMock)
    async def test_low_confidence_safe_escalates(self, mock_escalate):
        # Local model says safe, but confidence is low (0.92 < 0.98)
        mock_escalate.return_value = ReviewResult(
            verdict="safe",
            cwe="none",
            explanation="Checked by Gemini",
            source="gemini",
            suggested_fix=None,
        )
        reviewer = MockLocalReviewer(verdict="safe", confidence=0.92)
        result = await review_code("def foo(): pass", reviewer)

        self.assertEqual(result.source, "gemini")
        mock_escalate.assert_called_once()

    @patch("src.router.escalate_to_gemini", new_callable=AsyncMock)
    async def test_vulnerable_always_escalates(self, mock_escalate):
        # Local model flags vulnerable, even with high confidence
        mock_escalate.return_value = ReviewResult(
            verdict="vulnerable",
            cwe="CWE-89",
            explanation="SQL Injection confirmed by Gemini",
            source="gemini",
            suggested_fix="cursor.execute('...', (x,))",
        )
        reviewer = MockLocalReviewer(verdict="vulnerable", confidence=0.99, cwe="CWE-89")
        result = await review_code("select * from users", reviewer)

        self.assertEqual(result.source, "gemini")
        self.assertEqual(result.verdict, "vulnerable")
        self.assertEqual(result.suggested_fix, "cursor.execute('...', (x,))")
        mock_escalate.assert_called_once()

    @patch("src.router.escalate_to_gemini", new_callable=AsyncMock)
    async def test_unknown_verdict_escalates(self, mock_escalate):
        # Local model output could not be parsed
        mock_escalate.return_value = ReviewResult(
            verdict="safe",
            cwe="none",
            explanation="Gemini parsed successfully",
            source="gemini",
        )
        reviewer = MockLocalReviewer(verdict="unknown", confidence=0.0)
        result = await review_code("some code", reviewer)

        self.assertEqual(result.source, "gemini")
        mock_escalate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
