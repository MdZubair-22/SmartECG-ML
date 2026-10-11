import unittest

from backend.ecg_pipeline import summarize_predictions


class TestPredictionSummary(unittest.TestCase):
    def test_all_normal_predictions_are_normal(self):
        summary = summarize_predictions({"N": 12})

        self.assertEqual(summary["status"], "normal")
        self.assertEqual(summary["normal_beats"], 12)
        self.assertEqual(summary["abnormal_beats"], 0)

    def test_known_abnormal_predictions_take_precedence_over_unknown(self):
        summary = summarize_predictions({"N": 6, "Q": 2, "V": 1})

        self.assertEqual(summary["status"], "abnormal")
        self.assertEqual(summary["abnormal_beats"], 1)
        self.assertEqual(summary["inconclusive_beats"], 2)

    def test_unknown_predictions_alone_are_inconclusive(self):
        summary = summarize_predictions({"N": 4, "Q": 2})

        self.assertEqual(summary["status"], "inconclusive")
        self.assertEqual(summary["inconclusive_beats"], 2)

    def test_empty_distribution_is_inconclusive(self):
        summary = summarize_predictions({})

        self.assertEqual(summary["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
