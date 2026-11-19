import sys
import unittest
from unittest.mock import patch

import numpy as np

from general import GSASII_imports


@unittest.skipUnless(sys.platform == "win32", "Windows fmask binary required")
class TestFmask(unittest.TestCase):
    def test_mask_and_prevmask_accept_boolean_frame(self):
        import fmask

        image = np.array([10] * 10 + [1, 100, 100], dtype=np.int32)
        theta = np.full(image.shape, 0.5, dtype=np.float64)
        frame = np.zeros(image.shape, dtype=bool)
        frame[-1] = True
        bins = np.array([0.0, 1.0], dtype=np.float64)
        expected = np.array([False] * 10 + [True, True, False])

        for mask_function in (fmask.mask, fmask.prevmask):
            with self.subTest(function=mask_function.__name__):
                result = np.zeros_like(frame)
                count = mask_function(3.0, frame, theta, image, bins, result, 0.0, 1.0)
                self.assertEqual(count, 2)
                np.testing.assert_array_equal(result, expected)

    def test_generate_pixel_mask_uses_native_extension(self):
        image = np.full((4, 4), 10, dtype=np.int32)
        image[2, 2] = 100
        img = {
            "corrected_image": image,
            "Image Controls": {"IOtth": [0.1, 1.1], "wavelength": 0.5},
            "Masks": {"SpotMask": {}},
        }
        with patch.object(GSASII_imports, "GetDetectorXY2", return_value=(1, 1)):
            GSASII_imports.GeneratePixelMask(
                img,
                FrameMask=np.zeros_like(image, dtype=bool),
                ThetaMap=np.full(image.shape, 0.5, dtype=np.float64),
                numChans=2,
            )
        expected = np.zeros_like(image, dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(img["Masks"]["SpotMask"]["spotMask"], expected)
