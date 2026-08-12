import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReferenceJobDatabaseTests(unittest.TestCase):
    def test_tracks_the_two_user_uploaded_files(self):
        from platform_app.database import PlatformDatabase

        with tempfile.TemporaryDirectory() as directory:
            database = PlatformDatabase(Path(directory) / "platform.sqlite3")
            tenant = database.create_tenant("Test", "project_test", "token")
            job_id = database.create_reference_job(tenant["id"], "project_test", "photo.png", "standard.png")
            database.mark_reference_file_uploaded(job_id, "capture", "private/ref/capture.png")
            job = database.mark_reference_file_uploaded(job_id, "reference", "private/ref/reference.png")
            self.assertEqual(job["state"], "ready")
            self.assertEqual(job["capture_filename"], "photo.png")
            self.assertEqual(job["reference_filename"], "standard.png")


if __name__ == "__main__":
    unittest.main()
