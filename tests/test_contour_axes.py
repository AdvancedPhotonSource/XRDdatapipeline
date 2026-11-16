import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyqtgraph.Qt import QtWidgets

from general.corrections_and_maps import tth_to_q
from mainUI.contour import ContourView, Viewtype


class TestContourAxes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        integrals = os.path.join(self.tempdir.name, "integrals")
        os.mkdir(integrals)
        self.write_integral(integrals, "wide", np.linspace(1, 40, 4))
        self.write_integral(integrals, "narrow", np.linspace(1, 10, 5))
        self.settings = SimpleNamespace(
            output_directory=self.tempdir.name,
            keylist=["wide", "narrow"],
            curr_key=0,
            wavelength=0.5,
        )
        self.widget = ContourView(None, self.settings)
        self.addCleanup(self.widget.close)
        self.ctime = patch("mainUI.contour.os.path.getctime", return_value=0)
        self.ctime.start()
        self.addCleanup(self.ctime.stop)

    def write_integral(self, directory, name, xvals):
        path = os.path.join(directory, name + "00001_base.chi")
        with open(path, "w") as stream:
            stream.write("header\n" * 4)
            np.savetxt(stream, np.column_stack((xvals, np.arange(len(xvals)))))

    def test_dataset_switch_updates_contour_and_waterfall_axes(self):
        for manual in (False, True):
            for viewtype in (Viewtype.Contour, Viewtype.Waterfall):
                for axis in (0, 1):
                    with self.subTest(manual=manual, viewtype=viewtype, axis=axis):
                        self.widget.viewtype_select.setCurrentIndex(viewtype.value)
                        self.settings.curr_key = 0
                        self.widget.x_axis_type = axis
                        self.widget.update_dir()
                        self.settings.curr_key = 1
                        self.widget.reset_integral_data(manual=manual)

                        expected = np.linspace(1, 10, 5)
                        np.testing.assert_allclose(self.widget.tthvals, expected)
                        np.testing.assert_allclose(self.widget.qvals, tth_to_q(expected, 0.5))
                        np.testing.assert_array_equal(
                            self.widget.integral_data[0], np.arange(len(expected))
                        )
                        xvals = expected if axis == 0 else tth_to_q(expected, 0.5)
                        if viewtype == Viewtype.Contour:
                            rect = self.widget.contour_image.mapRectToParent(
                                self.widget.contour_image.boundingRect()
                            )
                            self.assertAlmostEqual(rect.left(), xvals[0])
                            self.assertAlmostEqual(rect.right(), xvals[-1])
                        else:
                            np.testing.assert_allclose(
                                self.widget.contour_waterfall[0].xData, xvals
                            )

    def test_empty_dataset_reloads_axis_when_integral_arrives(self):
        self.widget.update_dir()
        self.settings.keylist.append("new")
        self.settings.curr_key = 2
        self.widget.reset_integral_data()
        self.assertEqual(len(self.widget.tthvals), 0)
        self.assertEqual(len(self.widget.qvals), 0)

        expected = np.linspace(2, 12, 6)
        self.write_integral(
            os.path.join(self.tempdir.name, "integrals"), "new", expected
        )
        self.widget.update_integral_list()

        np.testing.assert_allclose(self.widget.tthvals, expected)
        np.testing.assert_allclose(self.widget.qvals, tth_to_q(expected, 0.5))
