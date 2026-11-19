import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pyqtgraph.Qt import QtWidgets

from pipeline.pipeline_UI import main_window


class TestSkipProcessed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.input_dir = Path(self.tempdir.name) / "input"
        self.input_dir.mkdir()
        self.output_dir = Path(self.tempdir.name) / "XRDdatapipeline_output"
        (self.output_dir / "integrals").mkdir(parents=True)
        self.window = main_window(
            input_directory=str(self.input_dir),
            output_directory=str(self.output_dir),
        )
        self.window.process_existing_only_radio.setChecked(True)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.timer.stop()
        logger = logging.getLogger()
        for handler in (self.window.ch, getattr(self.window, "fh", None)):
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()
        self.window.close()

    def create_image(self, name):
        (self.input_dir / (name + ".tif")).touch()

    def create_integral(self, name):
        (self.output_dir / "integrals" / (name + "_base.chi")).touch()

    def test_skip_existing_integrals_and_apply_filename_filters(self):
        for name in ("keep-00001", "keep_plain", "keep-new-00002",
                     "keep-ignore", "other"):
            self.create_image(name)
        self.create_integral("keep-00001")
        self.create_integral("keep_plain")
        self.window.settings_widget.regex_include_text.setText("keep")
        self.window.settings_widget.regex_exclude_text.setText("ignore")
        self.window.skip_processed_files.setChecked(True)

        self.window.start_processing()

        self.assertEqual(
            list(self.window.queue),
            [[os.path.join(str(self.input_dir), "keep-new-00002.tif"),
              "keep-new-00002", ".tif"]],
        )

    def test_process_existing_integrals_when_skip_is_unchecked(self):
        self.create_image("sample_00001")
        self.create_integral("sample_00001")

        self.window.start_processing()

        self.assertEqual([item[1] for item in self.window.queue], ["sample_00001"])

    def test_all_skipped_stops_without_a_cache_worker(self):
        self.create_image("sample_00001")
        self.create_integral("sample_00001")
        self.window.skip_processed_files.setChecked(True)

        self.window.start_button_pressed()
        self.window.on_timeout()

        self.assertFalse(self.window.keep_running)
        self.assertEqual(self.window.start_button.text(), "Start")
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertTrue(self.window.input_directory_widget.isEnabled())
        self.assertEqual(len(self.window.queue), 0)

    def test_stop_before_cache_started(self):
        self.create_image("sample_00001")
        self.window.start_button_pressed()

        self.window.stop_button_pressed()

        self.assertFalse(self.window.keep_running)
        self.assertEqual(self.window.start_button.text(), "Start")
        self.assertEqual(len(self.window.queue), 0)

    def test_stop_requests_running_cache_to_finish(self):
        self.create_image("sample_00001")
        self.window.start_button_pressed()
        worker = SimpleNamespace(stopEarly=False)
        thread = Mock()
        thread.isRunning.return_value = True
        self.window.cache_worker = worker
        self.window.cache_thread = thread

        self.window.stop_button_pressed()

        self.assertTrue(worker.stopEarly)
        thread.quit.assert_called_once()
        self.assertFalse(self.window.start_button.isEnabled())
