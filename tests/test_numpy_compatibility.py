import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from general import GSASII_imports as gsas


class TestNumpyCompatibility(unittest.TestCase):
    def test_mask_map_uses_float32_pixel_coordinates(self):
        data = {"pixelSize": [1000, 1000]}
        masks = {
            "Frames": [],
            "Polygons": [],
            "Points": np.array([[0.5], [0.5], [0.1]]),
        }

        result = gsas.MakeMaskMap(data, masks, (0, 2), (0, 2), None)

        np.testing.assert_array_equal(result, [[True, False], [False, False]])

    def test_two_theta_map_uses_float32_pixel_coordinates(self):
        data = {"pixelSize": [1000, 2000], "det2theta": 0, "PolaVal": [0.5]}

        def geometry(x, y, controls):
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.float32)
            np.testing.assert_array_equal(x, [[0.5, 1.5], [0.5, 1.5]])
            np.testing.assert_array_equal(y, [[1.0, 1.0], [3.0, 3.0]])
            return x, y, np.ones_like(x)

        with patch.object(gsas, "GetTthAzmG2", side_effect=geometry):
            result = gsas.Make2ThetaAzimuthMap(data, (0, 2), (0, 2))

        self.assertEqual(result.shape, (4, 2, 2))
        np.testing.assert_array_equal(result[0], [[0.5, 1.5], [0.5, 1.5]])

    def test_frame_mask_uses_float32_pixel_coordinates(self):
        data = {"pixelSize": [1000, 2000], "size": (2, 2)}

        def polymask(size, x, y, count, frame, output):
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.float32)
            np.testing.assert_array_equal(x, [0.5, 0.5, 1.5, 1.5])
            np.testing.assert_array_equal(y, [1.0, 3.0, 1.0, 3.0])
            return np.array([True, False, True, False])

        with patch.dict(sys.modules, {"polymask": SimpleNamespace(polymask=polymask)}):
            result = gsas.MakeFrameMask(data, [[0, 0], [1, 0], [0, 1]])

        np.testing.assert_array_equal(result, [[False, False], [True, True]])


if __name__ == "__main__":
    unittest.main()
