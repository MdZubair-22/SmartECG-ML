import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import image_service
from backend.production_digitizer import WEIGHT_PATHS, engine_config


class TestEcglightProductionDigitizer(unittest.TestCase):
    def test_env_config_resolves_ecglight_source_and_weights(self):
        root = Path(__file__).resolve().parents[1]
        repo = root / "third_party" / "ECGLight"
        python = Path(sys.executable).resolve()
        completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.dict(os.environ, {
            "SMART_ECG_ECGLIGHT_PYTHON": str(python),
            "SMART_ECG_ECGLIGHT_PATH": str(repo),
        }, clear=False), patch("backend.production_digitizer.subprocess.run", return_value=completed):
            config = engine_config()

        self.assertEqual(config["engine"], "ECGLight")
        self.assertEqual(config["repo"], str(repo.resolve()))
        self.assertEqual(config["python"], str(python))
        self.assertEqual(
            config["weights"],
            {name: str((repo / relative).resolve()) for name, relative in WEIGHT_PATHS.items()},
        )

    def test_fallback_can_be_disabled_as_preference_and_used_after_engine_failure(self):
        with patch.dict(os.environ, {
            "SMART_ECG_ALLOW_CLASSICAL_FALLBACK": "1",
            "SMART_ECG_PREFER_CLASSICAL_FALLBACK": "0",
        }, clear=False):
            with patch.object(image_service, "digitize_with_production_engine", side_effect=RuntimeError("weights missing")) as ecglight, \
                 patch.object(image_service, "digitize_ecg_image", return_value=(b"wfdb", {"detected_beats": 3})):
                wfdb_zip, metadata = image_service.digitize(b"image", "ecg.png")

        self.assertEqual(wfdb_zip, b"wfdb")
        self.assertEqual(metadata["engine"], "classical_fallback")
        self.assertEqual(metadata["fallback_reason"], "weights missing")
        ecglight.assert_called_once_with(b"image", "ecg.png")

    def test_classical_fallback_is_preferred_without_waiting_for_ecglight(self):
        with patch.dict(os.environ, {
            "SMART_ECG_ALLOW_CLASSICAL_FALLBACK": "1",
            "SMART_ECG_PREFER_CLASSICAL_FALLBACK": "1",
        }, clear=False):
            with patch.object(image_service, "digitize_with_production_engine") as ecglight, \
                 patch.object(image_service, "digitize_ecg_image", return_value=(b"wfdb", {"detected_beats": 3})):
                wfdb_zip, metadata = image_service.digitize(b"image", "ecg.png")

        self.assertEqual(wfdb_zip, b"wfdb")
        self.assertEqual(metadata["engine"], "classical_fallback")
        ecglight.assert_not_called()


if __name__ == "__main__":
    unittest.main()