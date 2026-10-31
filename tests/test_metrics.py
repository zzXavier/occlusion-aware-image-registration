import unittest

import numpy as np
from src.experiment import CFG, metrics, pair


class Metrics(unittest.TestCase):
    def test_identity_truth_has_zero_errors(self):
        _, _, truth, visible, valid = pair("moon.png", 94)
        values = metrics(truth, truth, visible, valid)
        np.testing.assert_allclose(values, 0)

    def test_split(self):
        self.assertFalse(set(CFG["train"]) & set(CFG["test"]))
