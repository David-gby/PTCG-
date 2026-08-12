from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_systemd_starts_v07_platform_with_a_persistent_workspace(self):
        service = (ROOT / "deployment" / "server" / "cardscope.service.template").read_text(encoding="utf-8")
        self.assertIn("platform_server.py", service)
        self.assertIn("/var/lib/cardscope/platform_workspace", service)
        self.assertNotIn("uvicorn", service)


if __name__ == "__main__":
    unittest.main()
