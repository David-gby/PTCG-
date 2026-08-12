import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RegistrationModeTests(unittest.TestCase):
    def test_requires_both_uploaded_images(self):
        from platform_app.reference_registration import (
            RegistrationInputError,
            validate_pair,
        )

        with self.assertRaisesRegex(RegistrationInputError, "REFERENCE_IMAGE_REQUIRED"):
            validate_pair(capture_png=b"capture", reference_png=None)


if __name__ == "__main__":
    unittest.main()
