import logging
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from pyqtgraph.Qt import QtCore, QtWidgets

import calibration_widget
from pipeline.pipeline_UI import main_window


class TestCalibrationImageDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_pipeline_passes_selected_directory_to_calibrator(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = main_window()
            self.addCleanup(pipeline.close)
            self.addCleanup(logging.getLogger().removeHandler, pipeline.ch)
            self.addCleanup(pipeline.ch.close)
            pipeline.input_directory_widget.file_name.setText(directory)
            process = Mock()
            mkstemp = tempfile.mkstemp
            with patch.object(QtCore, "QProcess", return_value=process), patch(
                "pipeline.pipeline_UI.tempfile.mkstemp",
                side_effect=lambda **kwargs: mkstemp(dir=directory, **kwargs),
            ):
                pipeline.open_calibration_program()

            executable, arguments = process.start.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(arguments[1], "--pipeline-handoff")
            self.assertEqual(arguments[3:], ["--image-directory", directory])

    def test_startup_parses_directory_and_existing_image_argument(self):
        argv = [
            "calibration_widget.py", "--pipeline-handoff", "handoff.txt",
            "--image-directory", "image folder", "sample.tif",
        ]
        app = Mock()
        app.exec.return_value = 0
        with patch.object(sys, "argv", argv), patch.object(
            calibration_widget.QtWidgets, "QApplication", return_value=app
        ), patch.object(calibration_widget, "GSAS2CalibratorApp") as window:
            self.assertEqual(calibration_widget.main(), 0)
        window.assert_called_once_with(
            pipeline_handoff_file="handoff.txt", image_directory="image folder"
        )
        window.return_value.load_image.assert_called_once_with("sample.tif")

    def test_standalone_startup_keeps_positional_image(self):
        app = Mock()
        app.exec.return_value = 0
        with patch.object(sys, "argv", ["calibration_widget.py", "sample.tif"]), patch.object(
            calibration_widget.QtWidgets, "QApplication", return_value=app
        ), patch.object(calibration_widget, "GSAS2CalibratorApp") as window:
            self.assertEqual(calibration_widget.main(), 0)
        window.assert_called_once_with(pipeline_handoff_file=None, image_directory=None)
        window.return_value.load_image.assert_called_once_with("sample.tif")

    def test_image_dialog_uses_directory_and_standalone_default(self):
        for directory in ("image folder", None):
            with self.subTest(directory=directory):
                window = calibration_widget.GSAS2CalibratorApp(image_directory=directory)
                self.addCleanup(window.close)
                with patch.object(
                    QtWidgets.QFileDialog, "getOpenFileName", return_value=("", "")
                ) as dialog:
                    window.btn_load_image.click()
                self.assertEqual(dialog.call_args.args[2], directory or "")
                self.assertEqual(
                    dialog.call_args.args[3], "TIFF files (*.tif *.tiff);;All files (*)"
                )
