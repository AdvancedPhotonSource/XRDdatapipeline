import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
from pyqtgraph.Qt import QtWidgets

from mask_widget import Arc, MainWindow, Ring, readMasks


class TestImmaskSpotImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        image_path = Path(self.tempdir.name) / "image.tif"
        tifffile.imwrite(image_path, np.full((512, 512), 100, dtype=np.int32))
        self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.addCleanup(self.window.file_select_window.close)
        self.window.file_select_window.close()
        self.window.load_image(str(image_path))
        self.window.cache = {
            "pixelSize": [1000, 1000],
            "pixelTAmap": np.zeros((512, 512)),
        }
        self.window.hasLoadedConfig = True
        self.window.imagescale_is_square = True
        self.app.processEvents()

    def write_mask(self, points):
        mask_path = Path(self.tempdir.name) / "spots.immask"
        mask_path.write_text(
            f"Points:{points}\n"
            "Rings:[]\nArcs:[]\nPolygons:[]\nFrames:[]\n"
            "Xlines:[]\nYlines:[]\nThresholds:[(0,200),[0,200]]\n"
        )
        return mask_path

    def test_imports_numpy_scalar_diameter_and_positions_preview(self):
        mask_path = self.write_mask(
            "[[201.07109476391582, 189.6424702365278, "
            "np.float64(24.89332027167271)], [45.0, 60.0, 20.0]]"
        )

        self.window.load_immask(str(mask_path))
        self.app.processEvents()

        first, second = self.window.main_image.objects
        np.testing.assert_allclose(
            first.center, [189.6424702365278, 201.07109476391582], atol=1e-6
        )
        self.assertAlmostEqual(first.radius, 24.89332027167271 / 2)
        np.testing.assert_allclose(second.center, [60.0, 45.0], atol=1e-6)
        self.assertAlmostEqual(second.radius, 10.0)

        self.window.preview_mask(override_checkbox=True)
        mask = self.window.main_image.predef_mask_data.mask_data
        self.assertTrue(mask[190, 201])
        self.assertTrue(mask[60, 45])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[190, 220])

    def test_import_preserves_non_square_pixel_conversion(self):
        self.window.cache["pixelSize"] = [1000, 2000]
        self.window.imagescale_is_square = False
        mask_path = self.write_mask("[[100.0, 120.0, np.float64(20.0)]]")

        self.window.load_immask(str(mask_path))
        self.app.processEvents()

        spot = self.window.main_image.objects[0]
        np.testing.assert_allclose(spot.center, [60.0, 100.0], atol=1e-6)
        self.assertAlmostEqual(spot.radius, 10.0 / np.sqrt(5))

    def test_import_example_with_150_micron_pixels(self):
        image_path = Path(self.tempdir.name) / "large_image.tif"
        tifffile.imwrite(image_path, np.full((2048, 2048), 100, dtype=np.int32))
        self.window.load_image(str(image_path))
        self.window.cache["pixelSize"] = [150, 150]
        mask_path = self.write_mask(
            "[[201.07109476391582, 189.6424702365278, np.float64(24.89332027167271)]]"
        )

        self.window.load_immask(str(mask_path))
        self.app.processEvents()

        spot = self.window.main_image.objects[0]
        np.testing.assert_allclose(
            spot.center, [189.6424702365278 / 0.15, 201.07109476391582 / 0.15],
            atol=1e-6,
        )
        self.assertAlmostEqual(spot.radius, 24.89332027167271 / 0.3)

    def test_invalid_radius_does_not_add_a_default_spot(self):
        mask_path = self.write_mask("[[100.0, 120.0, -20.0]]")

        with self.assertRaisesRegex(ValueError, "radius"):
            self.window.load_immask(str(mask_path))

        self.assertEqual(self.window.main_image.objects, [])


class TestImmaskAngularImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        y, x = np.mgrid[0:128, 0:128]
        self.tth = np.hypot(x - 64, y - 64) * 0.1
        self.azim = np.degrees(np.arctan2(y - 64, x - 64)) % 360
        for cls, attr, value in (
            (Ring, "tthmap", self.tth), (Ring, "azmap", self.azim),
            (Arc, "tthmap", self.tth), (Arc, "azmap", self.azim),
            (Arc, "center", [64, 64]),
        ):
            patcher = patch.object(cls, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        image_path = Path(self.tempdir.name) / "image.tif"
        tifffile.imwrite(image_path, np.full((128, 128), 100, dtype=np.int32))
        self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.addCleanup(self.window.file_select_window.close)
        self.window.file_select_window.close()
        self.window.load_image(str(image_path))
        self.window.cache = {"pixelSize": [1000, 1000], "pixelTAmap": self.tth}
        self.window.hasLoadedConfig = True
        self.window.imagescale_is_square = True
        self.window.max_tth_threshold.setMaximum(100)
        self.window.max_tth_threshold.setValue(100)
        self.app.processEvents()

    def load_mask(self, rings="[]", arcs="[]", repaint=True):
        mask_path = Path(self.tempdir.name) / "angular.immask"
        mask_path.write_text(
            f"Points:[]\nRings:{rings}\nArcs:{arcs}\nPolygons:[]\nFrames:[]\n"
            "Xlines:[]\nYlines:[]\nThresholds:[(0,200),[0,200]]\n"
        )
        self.window.load_immask(str(mask_path))
        if repaint:
            self.app.processEvents()
        self.window.preview_mask(override_checkbox=True)
        return self.window.main_image.objects[0]

    def test_ring_with_numpy_scalar_parameters(self):
        ring = self.load_mask(rings="[[np.float64(3.0), np.float64(1.0)]]", repaint=False)

        self.assertEqual(len(ring.handles), 2)
        expected = (self.tth >= 2.5) & (self.tth <= 3.5)
        np.testing.assert_array_equal(ring.mask_data, expected)
        self.app.processEvents()
        np.testing.assert_array_equal(ring.mask_data, expected)
        self.assertEqual((ring.tth_center, ring.tthwidth), (3.0, 1.0))
        self.assertTrue(ring.mask_data[64, 94])
        self.assertFalse(ring.mask_data[64, 104])
        np.testing.assert_array_equal(self.window.main_image.predef_mask_data.mask_data, expected)

    def test_arc_with_numpy_scalar_parameters(self):
        arc = self.load_mask(
            arcs="[[np.float64(3.0), [np.float64(30.0), np.float64(100.0)], np.float64(1.0)]]",
            repaint=False,
        )

        self.assertEqual(len(arc.handles), 5)
        expected = ((self.tth >= 2.5) & (self.tth <= 3.5)
                    & (self.azim >= 30) & (self.azim <= 100))
        np.testing.assert_array_equal(arc.mask_data, expected)
        np.testing.assert_array_equal(self.window.main_image.predef_mask_data.mask_data, expected)
        self.app.processEvents()
        np.testing.assert_array_equal(arc.mask_data, expected)
        self.assertEqual((arc.tth_center, arc.startazim, arc.endazim, arc.tthwidth),
                         (3.0, 30.0, 100.0, 1.0))
        self.assertTrue(arc.mask_data[89, 74])
        self.assertFalse(arc.mask_data[64, 94])
        self.assertTrue(self.window.main_image.predef_mask_data.mask_data[89, 74])

    def test_arc_crossing_zero_azimuth(self):
        arc = self.load_mask(arcs="[[3e0, [-1e1, 1e1], 1e0]]")

        expected = ((self.tth >= 2.5) & (self.tth <= 3.5)
                    & ((self.azim >= 350) | (self.azim <= 10)))
        np.testing.assert_array_equal(arc.mask_data, expected)
        self.assertTrue(arc.mask_data[64, 94])
        self.assertTrue(arc.mask_data[61, 94])
        self.assertFalse(arc.mask_data[64, 34])

    def test_full_circle_arc(self):
        arc = self.load_mask(arcs="[[3.0, [0.0, 360.0], 1.0]]")
        np.testing.assert_array_equal(arc.mask_data, (self.tth >= 2.5) & (self.tth <= 3.5))

    def test_dragging_imported_ring_updates_the_mask(self):
        ring = self.load_mask(rings="[[3.0, 1.0]]")
        original = ring.mask_data.copy()
        ring.handles[1]["item"].movePoint(ring.find_closest_point(4.0, 45))
        self.app.processEvents()
        self.assertFalse(np.array_equal(ring.mask_data, original))
        self.assertTrue(ring.mask_data[64, 101])

    def test_dragging_imported_arc_updates_the_mask(self):
        arc = self.load_mask(arcs="[[3.0, [30.0, 100.0], 1.0]]")
        original = arc.mask_data.copy()
        handle = next(h for h in arc.handles if h["name"] == "Azim_left")
        handle["item"].movePoint(arc.find_closest_point(3.0, 150))
        self.app.processEvents()
        self.assertFalse(np.array_equal(arc.mask_data, original))
        self.assertTrue(arc.mask_data[79, 38])

    def test_invalid_widths_do_not_add_default_objects(self):
        for kwargs in ({"rings": "[[3.0, -1.0]]"},
                       {"arcs": "[[3.0, [30.0, 100.0], -1.0]]"},
                       {"rings": "[[np.float64('nan'), 1.0]]"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "width"):
                    self.load_mask(**kwargs)
                self.assertEqual(self.window.main_image.objects, [])

    def test_update_from_table_accepts_scientific_notation(self):
        self.load_mask(rings="[[3.0, 1.0]]", arcs="[[3.0, [30.0, 100.0], 1.0]]")
        ring, arc = self.window.main_image.objects
        ring.table_item.setText("[3.1e0, 9e-1]")
        arc.table_item.setText("[3.1e0, [-1e1, 1e1], 9e-1]")
        self.window.update_objects_from_table()
        self.app.processEvents()

        self.assertEqual((ring.tth_center, ring.tthwidth), (3.1, 0.9))
        self.assertEqual((arc.tth_center, arc.startazim, arc.endazim, arc.tthwidth),
                         (3.1, 350.0, 10.0, 0.9))

    def test_ring_and_arc_save_and_reload(self):
        self.load_mask(rings="[[np.float64(3.123), np.float64(0.456)]]",
                       arcs="[[np.float64(3.234), [np.float64(30.5), np.float64(100.5)], np.float64(0.654)]]")
        output = Path(self.tempdir.name) / "saved.immask"
        with patch("mask_widget.get_save_file_location", return_value=str(output)):
            self.window.save_immask()
        saved = readMasks(str(output))
        self.assertEqual(saved["Rings"], [[3.123, 0.456]])
        self.assertEqual(saved["Arcs"], [[3.234, [30.5, 100.5], 0.654]])

        self.window.load_immask(str(output))
        self.app.processEvents()
        first_ring, first_arc, reloaded_ring, reloaded_arc = self.window.main_image.objects
        np.testing.assert_array_equal(reloaded_ring.mask_data, first_ring.mask_data)
        np.testing.assert_array_equal(reloaded_arc.mask_data, first_arc.mask_data)

    def test_saving_edited_angular_masks_uses_plain_numbers(self):
        self.load_mask(rings="[[3.0, 1.0]]", arcs="[[3.0, [30.0, 100.0], 1.0]]")
        ring, arc = self.window.main_image.objects
        ring.handles[1]["item"].movePoint(ring.find_closest_point(4.0, 45))
        handle = next(h for h in arc.handles if h["name"] == "Azim_left")
        handle["item"].movePoint(arc.find_closest_point(3.0, 150))
        self.app.processEvents()

        output = Path(self.tempdir.name) / "edited.immask"
        with patch("mask_widget.get_save_file_location", return_value=str(output)):
            self.window.save_immask()
        fields = dict(line.split(":", 1) for line in output.read_text().splitlines())
        self.assertEqual(len(ast.literal_eval(fields["Rings"])), 1)
        self.assertEqual(len(ast.literal_eval(fields["Arcs"])), 1)
