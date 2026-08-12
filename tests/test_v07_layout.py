from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V07LayoutTests(unittest.TestCase):
    def test_single_platform_entrypoint_and_canonical_backend_exist(self):
        self.assertTrue((ROOT / "platform_server.py").is_file())
        self.assertTrue((ROOT / "platform_app" / "service.py").is_file())
        self.assertTrue((ROOT / "ml_backend" / "ptcg_inference.py").is_file())
        self.assertTrue((ROOT / "web" / "enterprise.html").is_file())


if __name__ == "__main__":
    unittest.main()
