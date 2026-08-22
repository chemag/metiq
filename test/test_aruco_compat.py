#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, "..", "src"))

import aruco_common


class ArucoCompatibilityTest(unittest.TestCase):
    def test_normalizes_opencv_4_column_vector(self):
        ids = np.array([[2], [7], [0], [1]], dtype=np.int32)

        normalized = aruco_common.normalize_aruco_ids(ids)

        np.testing.assert_array_equal(normalized, [2, 7, 0, 1])
        self.assertEqual(normalized.shape, (4,))

    def test_normalizes_opencv_5_flat_vector(self):
        ids = np.array([2, 7, 0, 1], dtype=np.int32)

        normalized = aruco_common.normalize_aruco_ids(ids)

        np.testing.assert_array_equal(normalized, [2, 7, 0, 1])
        self.assertEqual(normalized.shape, (4,))

    def test_preserves_no_detection(self):
        self.assertIsNone(aruco_common.normalize_aruco_ids(None))


if __name__ == "__main__":
    unittest.main()
