import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from evaluate import ChannelBank, checked_subset, error_db, evaluate_case, gram, summarize, synthetic_estimate


def capacity(g, rho, subset, k):
    vals = [np.linalg.slogdet(np.eye(k) + rho / k * gf[np.ix_(subset, subset)])[1] / np.log(2.0) for gf in g]
    return np.mean(vals)


class TestDualCSI(unittest.TestCase):
    def test_noise_exact_and_reproducible(self):
        h = np.arange(1, 25).reshape(2, 3, 4).astype(complex) * (1 + 0.2j)
        for level in (-30, -20, -10):
            e = synthetic_estimate(h, level, 42, 7)
            self.assertAlmostEqual(error_db(h, e), level, places=7)
            np.testing.assert_array_equal(e, synthetic_estimate(h, level, 42, 7))
            self.assertFalse(np.array_equal(e, synthetic_estimate(h, level, 42, 8)))

    def test_selection_on_estimate_evaluation_on_truth(self):
        h = np.array([[[3., 1.]]], dtype=complex)
        estimate = np.array([[[0.1, 5.]]], dtype=complex)
        def greedy(g):
            np.testing.assert_allclose(g, gram(estimate), rtol=1e-6)
            return np.array([np.argmax(g[0].diagonal().real)])
        def fixed(g):
            np.testing.assert_allclose(g, gram(estimate), rtol=1e-6)
            return np.array([0])
        rows = evaluate_case(h, estimate, 1., 1, {'Capacity-Greedy': greedy, 'Model': fixed}, capacity)
        self.assertEqual(rows[0]['selected_ports'], '1')
        self.assertAlmostEqual(rows[0]['capacity'], 1.0)
        self.assertAlmostEqual(rows[0]['NCR'], 1.0)
        self.assertAlmostEqual(rows[1]['capacity'], np.log2(10))
        self.assertGreater(rows[1]['NCR'], 1.0)  # No artificial clipping.

    def test_perfect_rejected(self):
        h = np.ones((1, 2, 3), dtype=complex)
        with self.assertRaises(ValueError):
            error_db(h, h.copy())

    def test_invalid_subsets(self):
        for bad in ([0, 0], [0, 4], [0], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                checked_subset(bad, 2, 4)

    def test_site_macro_not_sample_average(self):
        rows = [dict(method='m', snr_db=0., site_id=s, NCR=x, capacity=x, greedy_capacity=1., actual_error_db=-20.)
                for s, x in [(0, .6), (0, .6), (0, .6), (1, 1.)]]
        self.assertAlmostEqual(summarize(rows)['m']['NCR'], .8)

    def test_bank_joins_by_id_not_row_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bank.h5'
            with h5py.File(path, 'w') as f:
                f['H_real'] = np.array([[[[1., 2.]]], [[[3., 4.]]]])
                f['H_imag'] = np.zeros((2, 1, 1, 2))
                f['base_sample_id'] = np.array([19, 7])
                f['site_id'] = np.array([2, 5])
            bank = ChannelBank(path, 'H_real', 'H_imag', 'base_sample_id')
            try:
                np.testing.assert_array_equal(bank.get(7, 5), [[[3., 4.]]])
                with self.assertRaises(ValueError):
                    bank.get(7, 2)
                with self.assertRaises(KeyError):
                    bank.get(0, 2)
            finally:
                bank.close()


if __name__ == '__main__':
    unittest.main()
