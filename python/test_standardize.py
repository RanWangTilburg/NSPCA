from unittest import TestCase
import unittest
import numpy as np
from nspcasolverimpl import standardize


class TestStandardize(TestCase):
    def test_standardized_2d(self):
        rep = 100
        similar = True
        thres = 0.001
        std = 2.0
        for i in range(0, rep):
            data = np.random.randn(100, 50)
            result = standardize(data, std)
            mean = 0
            var = 0

            for col in range(0, data.shape[1]):
                mean = np.mean(result[:, col])
                var = np.var(result[:, col])

                if np.abs(mean) > thres:
                    similar = False
                elif np.abs(var - std * std) > thres:
                    similar = False

        self.assertTrue(similar)

    def test_standardized_1d(self):
        rep = 100
        similar = True
        thres = 0.001
        std = 2.0
        for i in range(0, rep):
            data = np.random.randn(100)
            result = standardize(data, std)

            mean = np.mean(result)
            var = np.var(result)

            if np.abs(mean) > thres:
                similar = False
            elif np.abs(var - std * std) > thres:
                similar = False

        self.assertTrue(similar)

    def test_type_mismatch(self):
        with self.assertRaises(Exception) as context:
            data = 1
            result = standardize(data)
        self.assertTrue('Type Mismatch, must be 1d or 2d numpy array' in context.exception)

    def test_constant(self):
        with self.assertRaises(Exception) as context:
            data = np.full(100, 1)
            result = standardize(data)

        self.assertTrue('The input contains a constant column' in context.exception)

    def test_contain_inf(self):
        with self.assertRaises(Exception) as context:
            data = np.full(10, 10)
            data[1] = np.inf
            result = standardize(data)

        self.assertTrue('The input contains inf' in context.exception)

    def test_contain_na(self):
        with self.assertRaises(Exception) as context:
            data = np.full(10, 10)
            data[1] = np.nan
            result = standardize(data)

        self.assertTrue('The input contains NaN' in context.exception)

    def test_not_1d_or_2d(self):
        with self.assertRaises(Exception) as context:
            data = np.zeros((100, 100, 10))
            result = standardize(data)

        self.assertTrue('Input must be 1d or 2d array' in context.exception)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(TestStandardize))
