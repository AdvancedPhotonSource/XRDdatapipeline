import logging
import tempfile
import unittest
from unittest.mock import patch

from pyqtgraph.Qt import QtWidgets

from mask_widget import MainWindow
from pipeline.pipeline_UI import main_window


class TestMaskImageDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def check_file_dialogs(self, mask_window, expected_directory):
        self.addCleanup(mask_window.close)
        self.addCleanup(mask_window.file_select_window.close)
        with patch.object(
            QtWidgets.QFileDialog, "getOpenFileName", return_value=("", "")
        ) as dialog:
            mask_window.file_select_window.test_image_browse_button.click()
            self.assertEqual(dialog.call_args.args[2], expected_directory)
            with patch.object(mask_window, "load_image"):
                mask_window.load_image_button.click()
            self.assertEqual(dialog.call_args.args[2], expected_directory)
            for button in (
                mask_window.file_select_window.imctrl_file_browse_button,
                mask_window.load_imctrl_button,
            ):
                button.click()
                self.assertEqual(dialog.call_args.args[2], expected_directory)
                self.assertEqual(dialog.call_args.args[3], "Imctrl and PONI files (*.imctrl *.poni)")
            for button in (
                mask_window.file_select_window.immask_file_browse_button,
                mask_window.load_immask_button,
            ):
                button.click()
                self.assertEqual(dialog.call_args.args[2], expected_directory)
                self.assertEqual(dialog.call_args.args[3], "Immask files (*.immask)")

    def test_pipeline_uses_current_input_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = main_window()
            self.addCleanup(pipeline.close)
            self.addCleanup(logging.getLogger().removeHandler, pipeline.ch)
            self.addCleanup(pipeline.ch.close)
            pipeline.input_directory_widget.file_name.setText(directory)

            pipeline.open_maskwidget()

            self.check_file_dialogs(pipeline.mask_widget, directory)

    def test_standalone_defaults_to_current_directory(self):
        self.check_file_dialogs(MainWindow(), ".")

    def test_pipeline_without_input_defaults_to_current_directory(self):
        pipeline = main_window()
        self.addCleanup(pipeline.close)
        self.addCleanup(logging.getLogger().removeHandler, pipeline.ch)
        self.addCleanup(pipeline.ch.close)

        pipeline.open_maskwidget()

        self.check_file_dialogs(pipeline.mask_widget, ".")
