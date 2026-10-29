"""Input-integrity and legacy-protocol regressions. Run with unittest from repo root."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from eval import onset_metrics as metrics
from validation import load_ids, load_gh, load_benchmark, onset_times, validate_audio


class InputIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_duplicate_and_unsafe_ids_rejected(self):
        path = self.root / "test.txt"
        for text in ["a\na\n", "../outside\n", ""]:
            path.write_text(text)
            with self.assertRaises(ValueError): load_ids(path)

    def test_missing_annotation_is_not_an_empty_window(self):
        (self.root / "test.txt").write_text("present\nmissing\n")
        (self.root / "onsets").mkdir()
        (self.root / "onsets/present_times.txt").write_text("9.1\n")
        with self.assertRaises(FileNotFoundError): load_gh(self.root, expected_count=2)
        (self.root / "onsets/missing_times.txt").write_text("1.0\n")
        ids, refs, excluded = load_gh(self.root, expected_count=2)
        self.assertEqual(ids, ["present", "missing"])
        self.assertEqual(refs, {"missing": [1.0]})
        self.assertEqual(excluded[0]["clip_id"], "present")

    def test_audio_required_even_for_excluded_window(self):
        (self.root / "eligible.flac").touch()
        with self.assertRaises(ValueError): validate_audio(self.root, ["eligible", "empty"])
        (self.root / "empty.flac").touch()
        self.assertEqual(validate_audio(self.root, ["eligible", "empty"])["present_files"], 2)

    def test_bad_times_fail_without_normalizing_valid_legacy_order(self):
        for times in [["nan"], ["inf"], ["-0.1"]]:
            with self.assertRaises(ValueError): onset_times(times, "test")
        self.assertEqual(onset_times(["2", "1", "1"], "test"), [2., 1., 1.])

    def test_duplicate_benchmark_rows_fail(self):
        path = self.root / "benchmark.csv"
        path.write_text('video_id,query_label,onset_times\na,impact,1.0\na,impact,2.0\n')
        with self.assertRaises(ValueError): load_benchmark(path, expected_count=None)


class LegacyProtocol(unittest.TestCase):
    def test_detection_rate_and_non_exhaustive_nms_are_preserved(self):
        self.assertEqual(metrics.DETECT_SR, 22050)
        wave = np.zeros(7000)
        wave[1500] = 1.
        wave[3500] = .5
        # Legacy list mutation skips the second suppressed candidate. An exhaustive
        # rewrite would return only2000 and fail this compatibility regression.
        self.assertEqual(metrics.onset_nms(np.array([2000,3000,4000]), wave, window=.12).tolist(),
                         [2000,4000])

    def test_match_boundary_is_strict_and_missing_audio_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                metrics.score(root, {"missing": [1.]}, .3, rise=False)
            (root / "clip.flac").touch()
            with patch.object(metrics, "detect_onsets", return_value=(np.array([0]), np.ones(1200))), \
                 patch.object(metrics, "onset_nms", return_value=np.array([0])):
                result = metrics.score(root, {"clip": [.05]}, .3, rise=False)
            self.assertEqual(result["f1@50"], 0.)
            self.assertEqual(result["f1@100"], 1.)


if __name__ == "__main__":
    unittest.main()
