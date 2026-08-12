from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReferenceUiContractTests(unittest.TestCase):
    def test_reference_mode_has_two_user_uploads_and_a_separate_mode_switch(self):
        page = (ROOT / "web" / "enterprise.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "enterprise.js").read_text(encoding="utf-8")
        for identifier in ("traditionalModeButton", "referenceModeButton", "referenceCaptureInput", "referenceStandardInput"):
            self.assertIn(f'id="{identifier}"', page)
        self.assertIn('id="detectionModeLabel"', page)
        self.assertIn('id="exportResultJson"', page)
        self.assertIn("/reference-inspections", script)
        self.assertIn("selectDetectionMode", script)
        self.assertIn("参考图配准（标准图对比）", script)
        self.assertIn("result_json_url", script)


if __name__ == "__main__":
    unittest.main()
