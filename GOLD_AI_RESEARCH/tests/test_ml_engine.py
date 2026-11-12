import unittest

import numpy as np
import pandas as pd

from gold_ai.ml_engine import _chronological_sets, _contiguous_forward_mask, train_model


class MlPipelineTests(unittest.TestCase):
    def test_splits_are_chronological_and_embargo_forward_labels(self):
        index = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
        features = pd.DataFrame({"feature": np.arange(100)}, index=index)
        targets = pd.Series(np.tile([0, 1], 50), index=index, name="target")
        train, validation, test = _chronological_sets(features, targets, 3, 0.6, 0.2)
        self.assertLess(train.index[-1] + pd.Timedelta(minutes=45), validation.index[0])
        self.assertLess(validation.index[-1] + pd.Timedelta(minutes=45), test.index[0])
        self.assertLess(train.index[-1], validation.index[0])
        self.assertLess(validation.index[-1], test.index[0])

    def test_synthetic_label_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, "only.*real"):
            train_model(pd.DataFrame(), "unused.joblib", data_label="SYNTHETIC")

    def test_future_labels_do_not_cross_missing_bar_gaps(self):
        index = pd.DatetimeIndex([
            *pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC"),
            *pd.date_range("2026-01-01 02:00", periods=5, freq="15min", tz="UTC"),
        ])
        mask = _contiguous_forward_mask(index, 2)
        self.assertFalse(mask.iloc[3])
        self.assertTrue(mask.iloc[0])


if __name__ == "__main__":
    unittest.main()
