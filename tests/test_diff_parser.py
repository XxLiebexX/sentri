import os
import unittest
import tempfile
from pathlib import Path

from src.diff_parser import (
    parse_diff,
    slice_spec_for_file,
    extract_function_context,
    build_chunks,
    CodeChunk,
)


class TestDiffParser(unittest.TestCase):
    def test_parse_diff_single_hunk(self):
        diff_text = """diff --git a/src/app.py b/src/app.py
index abc..def 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,4 +10,5 @@ def get_user():
     user = fetch()
+    validate(user)
     return user
"""
        hunks = parse_diff(diff_text)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0]["file"], "src/app.py")
        self.assertEqual(hunks[0]["line_start"], 10)
        self.assertEqual(hunks[0]["line_end"], 14)
        self.assertIn("validate(user)", hunks[0]["diff_text"])

    def test_parse_diff_multiple_files(self):
        diff_text = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,3 +1,4 @@
 def a():
+    pass
     return 1
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -20,2 +20,3 @@ def b():
+    x = 2
     return x
"""
        hunks = parse_diff(diff_text)
        self.assertEqual(len(hunks), 2)
        self.assertEqual(hunks[0]["file"], "src/a.py")
        self.assertEqual(hunks[1]["file"], "src/b.py")

    def test_slice_spec_for_file_matching(self):
        mock_spec = {
            "project_type": "FastAPI Web Service",
            "purpose": "User management API",
            "key_modules": [
                {"path": "backend/app/api", "role": "Routing and HTTP handlers"},
                {"path": "backend/app/db", "role": "Database connections"},
            ],
            "conventions": ["Use Pydantic for validation", "Async route handlers"],
            "risk_areas": ["Raw database queries", "Auth headers"],
        }

        # Slicing for api file
        api_slice = slice_spec_for_file(mock_spec, "backend/app/api/users.py")
        self.assertIn("FastAPI Web Service", api_slice)
        self.assertIn("Routing and HTTP handlers", api_slice)
        self.assertIn("Use Pydantic for validation", api_slice)
        self.assertIn("Raw database queries", api_slice)

        # Slicing for db file
        db_slice = slice_spec_for_file(mock_spec, "backend/app/db/session.py")
        self.assertIn("Database connections", db_slice)
        self.assertNotIn("Routing and HTTP handlers", db_slice)

    def test_slice_spec_empty(self):
        self.assertEqual(slice_spec_for_file(None, "foo.py"), "")
        self.assertEqual(slice_spec_for_file({}, "foo.py"), "")

    def test_extract_function_context_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_code = """import os

def helper():
    return 42

def target_function(x, y):
    val = x + y
    if val > 10:
        val = 10
    return val

def another():
    pass
"""
            py_file = Path(tmpdir) / "test_module.py"
            py_file.write_text(sample_code)

            # Target lines inside target_function (lines 7 to 9)
            context = extract_function_context("test_module.py", 7, 9, repo_path=tmpdir)
            self.assertIn("def target_function(x, y):", context)
            self.assertIn("return val", context)
            self.assertNotIn("def helper():", context)
            self.assertNotIn("def another():", context)

    def test_build_chunks_with_spec_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "service.py"
            py_file.write_text("def process():\n    return True\n")

            diff = """--- a/service.py
+++ b/service.py
@@ -1,2 +1,3 @@
 def process():
+    validate()
     return True
"""
            spec = {
                "project_type": "Data Pipeline",
                "purpose": "Batch ETL",
                "key_modules": [{"path": "service.py", "role": "Main processor"}],
                "conventions": ["Always validate inputs"],
                "risk_areas": ["Data corruption"],
            }

            chunks = build_chunks(diff, repo_path=tmpdir, spec=spec)
            self.assertEqual(len(chunks), 1)
            chunk = chunks[0]
            self.assertEqual(chunk.file, "service.py")
            self.assertIn("Data Pipeline", chunk.spec_context)
            self.assertIn("Main processor", chunk.spec_context)
            self.assertIn("Always validate inputs", chunk.spec_context)


if __name__ == "__main__":
    unittest.main()
