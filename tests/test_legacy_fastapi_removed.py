from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LegacyFastApiRemovalTests(unittest.TestCase):
    def test_v07_platform_is_the_only_service_entrypoint(self):
        self.assertTrue((ROOT / "platform_server.py").is_file())
        self.assertFalse((ROOT / "app.py").exists())
        self.assertFalse((ROOT / "api_pipeline.py").exists())
        self.assertFalse((ROOT / "card_centering").exists())
        self.assertFalse((ROOT / "steps").exists())


if __name__ == "__main__":
    unittest.main()
