import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import tifffile
from pyqtgraph.Qt import QtCore

from general.corrections_and_maps import tth_to_d, tth_to_q
from mainUI.main_image import MainImageView
from mainUI.main_window import KeyPressWindow


class TestMainImageMaps(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.maps_dir = os.path.join(self.tempdir.name, "maps")
        os.mkdir(self.maps_dir)
        self.view = SimpleNamespace(
            settings=SimpleNamespace(
                output_directory=self.tempdir.name,
                imagecontrol=os.path.join(self.tempdir.name, "selected.imctrl"),
                image_size=(2, 3),
            ),
            maps_loaded=False,
        )

    def write_map(self, name, value, shape=(2, 3)):
        tifffile.imwrite(
            os.path.join(self.maps_dir, name),
            np.full(shape, value, dtype=np.float32),
        )

    def test_loads_selected_calibration_maps_not_first_available(self):
        self.write_map("other_2thetamap.tif", 11)
        self.write_map("other_azmmap.tif", 12)
        self.write_map("selected_2thetamap.tif", 21)
        self.write_map("selected_azmmap.tif", 22)

        MainImageView.load_maps(self.view)

        self.assertTrue(self.view.maps_loaded)
        np.testing.assert_array_equal(self.view.tth_map, np.full((2, 3), 21))
        np.testing.assert_array_equal(self.view.azim_map, np.full((2, 3), 22))

        self.view.settings.imagecontrol = os.path.join(self.tempdir.name, "other.imctrl")
        MainImageView.load_maps(self.view)
        np.testing.assert_array_equal(self.view.tth_map, np.full((2, 3), 11))
        np.testing.assert_array_equal(self.view.azim_map, np.full((2, 3), 12))

    def test_missing_selected_map_does_not_use_another_calibration(self):
        self.write_map("selected_2thetamap.tif", 21)
        self.write_map("other_2thetamap.tif", 11)
        self.write_map("other_azmmap.tif", 12)
        self.view.maps_loaded = True

        MainImageView.load_maps(self.view)

        self.assertFalse(self.view.maps_loaded)

    def test_rejects_maps_with_different_image_shape(self):
        self.write_map("selected_2thetamap.tif", 21)
        self.write_map("selected_azmmap.tif", 22, shape=(3, 2))
        self.view.maps_loaded = True

        MainImageView.load_maps(self.view)

        self.assertFalse(self.view.maps_loaded)

    def test_poni_uses_matching_maps(self):
        self.view.settings.imagecontrol = os.path.join(self.tempdir.name, "selected.poni")
        self.write_map("selected_2thetamap.tif", 21)
        self.write_map("selected_azmmap.tif", 22)

        MainImageView.load_maps(self.view)

        self.assertTrue(self.view.maps_loaded)
        np.testing.assert_array_equal(self.view.tth_map, np.full((2, 3), 21))

    def test_readout_uses_selected_map_and_wavelength(self):
        self.write_map("other_2thetamap.tif", 11)
        self.write_map("other_azmmap.tif", 12)
        self.write_map("selected_2thetamap.tif", 21)
        self.write_map("selected_azmmap.tif", 22)
        MainImageView.load_maps(self.view)
        self.view.image_data = np.ones((2, 3))
        self.view.view = Mock()
        self.view.view.sceneBoundingRect.return_value.contains.return_value = True
        self.view.view.vb.mapSceneToView.return_value = QtCore.QPointF(1, 1)
        window = SimpleNamespace(
            imageview=self.view,
            settings=SimpleNamespace(wavelength=0.5),
            cursor_label=Mock(),
            tth_tracking_active=True,
            update_tth_lines=Mock(),
        )

        KeyPressWindow.mouseMovedImage(window, object())

        window.update_tth_lines.assert_called_once_with(
            21, tth_to_q(21, 0.5), tth_to_d(21, 0.5),
            update_image_location_text=False,
        )
        text = window.cursor_label.setText.call_args.args[0]
        self.assertIn("21.00", text)
        self.assertIn("22.0", text)
        self.assertIn(f"Q={tth_to_q(21, 0.5):.2f}", text)
        self.assertIn(f"d={tth_to_d(21, 0.5):.2f}", text)

    def test_missing_maps_shows_unavailable_and_does_not_track(self):
        self.view.image_data = np.ones((2, 3))
        self.view.view = Mock()
        self.view.view.sceneBoundingRect.return_value.contains.return_value = True
        self.view.view.vb.mapSceneToView.return_value = QtCore.QPointF(1, 1)
        window = SimpleNamespace(
            imageview=self.view,
            cursor_label=Mock(),
            tth_tracking_active=True,
            update_tth_lines=Mock(),
        )

        with patch("mainUI.main_window.QtWidgets.QMessageBox.warning") as warning:
            KeyPressWindow.warn_missing_maps(window)
        warning.assert_called_once()
        KeyPressWindow.mouseMovedImage(window, object())

        self.assertIn("unavailable", window.cursor_label.setText.call_args.args[0])
        window.update_tth_lines.assert_not_called()
