import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from src.summarize import ProjectSpec
import src.cache as cache_module


class TestCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sentri_cache_test_")
        self.patcher = patch.object(cache_module, "CACHE_DIR", Path(self.test_dir))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_retrieve_spec(self):
        spec = ProjectSpec(
            project_type="CLI Tool",
            purpose="System benchmarking",
            architecture="Modular CLI",
            key_modules=[{"path": "bench.py", "role": "Runner"}],
            conventions=["PEP 8"],
            risk_areas=["Memory usage"],
            repo_url="https://github.com/example/bench",
            commit_hash="abc12345",
        )

        cache_module.save_spec(spec)

        # Retrieve with matching commit
        cached = cache_module.get_cached_spec("https://github.com/example/bench", "abc12345")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.project_type, "CLI Tool")
        self.assertEqual(cached.commit_hash, "abc12345")
        self.assertEqual(len(cached.key_modules), 1)

    def test_cache_miss_on_new_commit(self):
        spec = ProjectSpec(
            project_type="Web App",
            purpose="Demo app",
            architecture="MVC",
            key_modules=[],
            conventions=[],
            risk_areas=[],
            repo_url="https://github.com/example/app",
            commit_hash="commit_v1",
        )
        cache_module.save_spec(spec)

        # Try to retrieve with a newer commit
        stale = cache_module.get_cached_spec("https://github.com/example/app", "commit_v2")
        self.assertIsNone(stale)

    def test_cache_miss_on_unknown_repo(self):
        result = cache_module.get_cached_spec("https://github.com/unknown/repo", "any_hash")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
