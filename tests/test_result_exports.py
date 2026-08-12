import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResultExportTests(unittest.TestCase):
    def test_writes_json_without_an_absolute_workspace_path(self):
        from platform_app.result_exports import inspection_result_path, write_inspection_result

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            inspection = {
                "id": "ins_123",
                "prediction": {"measurement_mode": "traditional"},
                "images": {"preview": "/api/platform/v1/inspections/ins_123/image"},
            }
            path = write_inspection_result(workspace, inspection)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path, inspection_result_path(workspace, "ins_123"))
            self.assertNotIn(str(workspace), json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
