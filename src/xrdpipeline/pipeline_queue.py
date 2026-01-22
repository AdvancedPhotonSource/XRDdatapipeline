"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the UI and base functionality of the analysis pipeline.
"""

from collections import deque
import argparse
import glob
import re
import os, sys
import subprocess
import time
import threading
import logging

from PIL import Image

import PySide6
from pyqtgraph.Qt import QtCore, QtWidgets

from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer

from general.GSASII_imports import *
from pipeline.pipeline import run_iteration
from pipeline.cache_creation import getmaps, get_azimbands, prepare_integration_maps, gradient_cache, create_cache
from general.corrections_and_maps import get_Qbands
from mask_widget import MainWindow
from general.file_selection import FileSelectRowWidget

class ImageMonitor(RegexMatchingEventHandler):
    """
    Watches for new images coming in
    """
    def __init__(self, queue, include=None, exclude=None):
        # dir\name_number_ext.tif or dir\name-number_ext.tif
        #'number' may be 00000 or xxxxx_xxxxx or xxxxx-xxxxx
        #'_ext' not on base images
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif.metadata$"
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif$"
        reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
        # reg for integral data files

        # RegexMatchingEventHandler uses the OR of all passed entries, not AND, so it must be built into the string
        if (include is not None) and (include.strip() != ""):
            reg_include = r"(?P<input_directory>.*[\\\/])(?P<name>.*" + re.escape(include) + r".*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
            regs = [reg_include]
        else:
            regs = [reg_image]
        ignore_regs = None
        if (exclude is not None) and (exclude.strip() != ""):
            ignore_regs = [r".*" + re.escape(exclude) + r".*"]
        RegexMatchingEventHandler.__init__(self, regexes=regs, ignore_regexes=ignore_regs)
        self.queue = queue

    def on_created(self, event):
        print("New file at {0}".format(event.src_path))
        results = [r.match(event.src_path) for r in self.regexes]
        # print(results[0].group(0,1,2,3,4))
        print(
            "Directory: {0}, Name: {1}, Number: {2}, Extension: {3}".format(
                results[0].group("input_directory"),
                results[0].group("name"),
                results[0].group("number"),
                results[0].group("ext"),
            )
        )
        # number -> actual int
        # number_int = results[0].group("number").remove("-").remove("_")
        # number_int = int(number_int)

        # Add file path to queue, stripping ".metadata"
        # TODO: WARNING: REMOVE THE 10X FOR ACTUAL RUNS
        # Done for testing 10k input without requiring 10x disk space
        # for i in range(10000):
        #    self.queue.append([event.src_path[:-9],results[0].group("name"),results[0].group("number")])
        # self.queue.append([event.src_path[:-9],results[0].group("name"),results[0].group("number")])
        self.queue.append(
            [
                event.src_path,
                results[0].group("name"),
                results[0].group("number"),
                results[0].group("ext"),
            ]
        )
        # self.queue.put([event.src_path,results.group("name"),results.group("number")])


class ImctrlFileSelect(FileSelectRowWidget):
    """
    Image control file selection dialog. Filters shown files to what is given in ext,
    and emits a signal when a file is selected.
    This signal is slotted in to the UI to read in a few modifiable values.
    """
    imctrl_set = QtCore.Signal()

    def __init__(self, label, default_text=None, startdir=".", ext=None):
        super().__init__(label=label, default_text=default_text, isdir=False, startdir=startdir, ext=ext)

    def select_file(self):
        location = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select File", self.startdir, self.ext
        )
        self.file_name.setText(location[0])
        self.imctrl_set.emit()


class CacheCreator(QtCore.QObject):
    """
    QObject which will run the cache creation routine in its own QThread.

    :param cache: Dictionary to hold the cached information
    :param input_directory: Location of the detector image files
    :param output_directory: Location to output files from the pipeline
    :param filename: Name of the first image file
    :param imctrlname: Name of the image control file
    :param flatfield: Name of the flatfield correction file, if any
    :param imgmaskname: Name of the predefined experimental mask file, if any
    :param bad_pixels: Name of the detector bad pixel mask file, if any
    :param blkSize: Block size
    :param calc_outlier: Calculate the outlier mask for each image and integrate using it
    :param esdMul: Multiplier of median absolute deviation to be used to determine outliers
    in each band
    :param outChannels: Number of integration bins
    :param calc_splitting: Calculate spot/texture classification and integrate using the
    separated masks
    :param azim_Q_shape_min: Minimum ratio of azimuthal to Q width for a cluster to be
    considered texture
    :param tth_integration_range: 2theta integration range in the form [min, max], if this should be changed from the value in the image control file
    :param azim_integration_range: Azimuthal integration range in the form [min, max], if this should be changed from the value in the image control file
    :param n_integration_bins: Number of integration bins, if this should be changed from the value in the image control file
    :param polarization: Polarization of the image, if this should be changed from the value in the image control file
    :param logging: Report timing information
    """
    cache_location_signal = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        cache,
        input_directory,
        output_directory,
        filename,
        imctrlname,
        flatfield,
        imgmaskname,
        bad_pixels,
        blkSize,
        esdMul = 3.0,
        outChannels = None,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        tth_integration_range=None,
        azim_integration_range=None,
        n_integration_bins=None,
        polarization=None,
        logging=False,
    ):
        super().__init__()
        self.cache = cache
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.filename = filename
        self.imctrlname = imctrlname
        self.flatfield = flatfield
        self.imgmaskname = imgmaskname
        self.bad_pixels = bad_pixels
        self.blkSize = blkSize
        self.logging = logging
        self.esdMul = esdMul
        self.outChannels = outChannels
        self.calc_splitting = calc_splitting
        self.azim_Q_shape_min = azim_Q_shape_min
        self.tth_integration_range=tth_integration_range
        self.azim_integration_range=azim_integration_range
        self.n_integration_bins=n_integration_bins
        self.polarization=polarization
        self.stopEarly = False

    def run(self):
        cache_time = time.time()
        cache_location = create_cache(
            self.cache,
            self.filename,
            self.imctrlname,
            self.output_directory,
            self.tth_integration_range,
            self.azim_integration_range,
            self.n_integration_bins,
            self.polarization,
            self.imgmaskname,
            self.bad_pixels,
            self.flatfield,
            self.esdMul,
            verbose=False,
        )

        cache_time = time.time() - cache_time
        print(f"Cache completed in {cache_time:.2f}s.")

        self.cache_location_signal.emit(cache_location)
        self.finished.emit()


class SingleIterator(QtCore.QObject):
    """
    QObject which will process a single image in its own QThread.
    Emits a signal when finished.

    :param cache: Dictionary of cached information usable by all images
    :param filename: Name of the image file to process
    :param imctrlname: Name of the image control file
    :param imgmaskname: Name of the predefined experimental mask file
    :param input_directory: Directory holding the detector image files
    :param output_directory: Directory to output files from the pipeline
    :param name: Name of the dataset
    :param number: Number of this image
    :param ext: Image file extension
    :param closing_method: Method used to expand the outlier mask. Default is
    binary closing.
    :param calc_outlier: Whether to calculate the outlier mask and integrate
    using it
    :param calc_splitting: Whether to calculate spot/texture classification and
    integrate using the separated masks
    :param azim_Q_shape_min: Minimum ratio of azimuthal width to Q width for a
    cluster to be considered texture
    :param calc_spot_stats: Whether to calculate basic area, number, etc. statistics
    on spot-tagged clusters. Adds <.1s to processing time.
    :param calc_grad_spottiness: Whether to calculate mean, standard deviation, and
    other statistics on each bin of the second azimuthal derivative. Adds 1-2s to
    processing time.
    :param csim_first_index: Index number for which image should be considered the first
    in a dataset for cosine similarity comparison.
    :param timing: Whether to return timing information for each step
    :param timing_names: List of names to append to for each timing checkpoint
    """
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    succeeded = QtCore.Signal()
    failed = QtCore.Signal()

    def __init__(
        self,
        filename,
        imctrlname,
        imgmaskname,
        input_directory,
        output_directory,
        name,
        number,
        ext,
        cache_location = None,
        closing_method="binary_closing",
        calc_outlier = True,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        calc_spot_stats = True,
        calc_grad_spottiness = False,
        csim_first_index = 0,
        timing=None,
        timing_names = None,
    ):
        super().__init__()
        self.filename = filename
        self.imctrlname = imctrlname
        self.imgmaskname = imgmaskname
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.name = name
        self.number = number
        self.ext = ext
        self.cache_location = cache_location
        self.closing_method = closing_method
        self.calc_outlier = calc_outlier
        self.calc_splitting = calc_splitting
        self.azim_Q_shape_min = azim_Q_shape_min
        self.calc_spot_stats = calc_spot_stats
        self.calc_grad_spottiness = calc_grad_spottiness
        self.csim_first_index = csim_first_index
        self.timing = timing
        self.timing_names = timing_names

    def run(self):
        try:
            run_iteration(
                self.filename,
                self.input_directory,
                self.output_directory,
                self.name,
                self.number,
                self.cache_location,
                self.ext,
                calc_outlier = self.calc_outlier,
                calc_splitting = self.calc_splitting,
                azim_Q_shape_min = self.azim_Q_shape_min,
                calc_spot_stats = self.calc_spot_stats,
                calc_grad_spottiness = self.calc_grad_spottiness,
                csim_first_index = self.csim_first_index,
                timing = self.timing,
                timing_names = self.timing_names,
            )
            self.succeeded.emit()
        except:
            logging.getLogger('').exception(f"Exception in file {self.filename}")
            self.failed.emit()
        self.finished.emit()


class AdvancedSettings(QtWidgets.QWidget):
    """
    Subwidget holding the advanced settings section of the UI
    """
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

        # self.settings_label = QtWidgets.QLabel("Advanced Settings")
        # self.override_config_label = QtWidgets.QLabel("Override Config Values: ")
        self.override_label = QtWidgets.QLabel("Override configuration file values by checking the box and setting the value.")

        self.madmult_override = QtWidgets.QCheckBox("Multiple of median absolute deviation for outlier masking:")
        self.madmult_override_default = False
        self.madmult_override.setChecked(self.madmult_override_default)
        self.madmult = QtWidgets.QDoubleSpinBox()
        self.madmult_default = 3
        self.madmult.setMinimum(0)
        self.madmult.setMaximum(10)
        self.madmult.setSingleStep(0.1)
        self.madmult.setValue(self.madmult_default)
        # self.madmult_label.setDisabled(True)
        # self.madmult.setDisabled(True)
        self.nbins_om_override = QtWidgets.QCheckBox("Number of 2theta bins for outlier masking:")
        self.nbins_om_override_default = False
        self.nbins_om_override.setChecked(self.nbins_om_override_default)
        self.nbins_om = QtWidgets.QSpinBox()
        self.nbins_om_default = 1000
        self.nbins_om.setMinimum(0)
        self.nbins_om.setMaximum(10000)
        self.nbins_om.setValue(self.nbins_om_default)
        self.azim_q_override = QtWidgets.QCheckBox("Azim / Q classification ratio:")
        self.azim_q_override_default = False
        self.azim_q_override.setChecked(self.azim_q_override_default)
        self.azim_q = QtWidgets.QSpinBox()
        self.azim_q_default = 100
        self.azim_q.setMinimum(0)
        self.azim_q.setMaximum(1000)
        self.azim_q.setValue(self.azim_q_default)
        self.csim_first_label = QtWidgets.QLabel("Cosine Similarity: First image index for comparison")
        self.csim_first_default = 0
        self.csim_first_spinbox = QtWidgets.QSpinBox()
        self.csim_first_spinbox.setValue(self.csim_first_default)

        self.calc_outlier_checkbox = QtWidgets.QCheckBox("Perform outlier masking")
        self.calc_outlier_default = True
        self.calc_outlier_checkbox.setChecked(self.calc_outlier_default)
        self.calc_outlier_checkbox.checkStateChanged.connect(self.toggle_outlier_settings)
        self.calc_splitting_checkbox = QtWidgets.QCheckBox("Perform spot/texture outlier mask splitting")
        self.calc_splitting_default = True
        self.calc_splitting_checkbox.setChecked(self.calc_splitting_default)
        self.calc_spottiness_label = QtWidgets.QLabel("Calculate Spottiness of Rings")
        self.calc_spottiness_combobox = QtWidgets.QComboBox()
        self.calc_spottiness_types = [
            "None",
            "Spot Area Stats Only",
            "Spot Area Stats and Gradient Statistics",
        ]
        self.calc_spottiness_combobox.addItems(self.calc_spottiness_types)
        self.calc_spottiness_default = 1
        self.calc_spottiness_combobox.setCurrentIndex(self.calc_spottiness_default)

        self.regex_include_label = QtWidgets.QLabel("Only include filenames with:")
        self.regex_include_text = QtWidgets.QLineEdit()
        self.regex_exclude_label = QtWidgets.QLabel("Exclude filenames with:")
        self.regex_exclude_text = QtWidgets.QLineEdit()

        self.defaults_button = QtWidgets.QPushButton("Restore Defaults")
        self.defaults_button.released.connect(self.restore_defaults)

        self.outlier_settings = QtWidgets.QWidget()
        self.outlier_layout = QtWidgets.QGridLayout()
        self.outlier_settings.setLayout(self.outlier_layout)
        self.settings_layout = QtWidgets.QGridLayout()
        # self.settings_layout.addWidget(self.settings_label, 0, 0, 1, 2)
        self.settings_layout.addWidget(self.regex_include_label, 0, 0)
        self.settings_layout.addWidget(self.regex_include_text, 0, 1)
        self.settings_layout.addWidget(self.regex_exclude_label, 1, 0)
        self.settings_layout.addWidget(self.regex_exclude_text, 1, 1)
        self.settings_layout.addWidget(self.csim_first_label, 2, 0)
        self.settings_layout.addWidget(self.csim_first_spinbox, 2, 1)
        self.settings_layout.addWidget(self.calc_outlier_checkbox, 3, 0, 1, 2)
        self.outlier_layout.addWidget(self.override_label, 0, 0, 1, 2)
        self.outlier_layout.addWidget(self.madmult_override, 1, 0)
        self.outlier_layout.addWidget(self.madmult, 1, 1)
        self.outlier_layout.addWidget(self.nbins_om_override, 2, 0)
        self.outlier_layout.addWidget(self.nbins_om, 2, 1)
        self.outlier_layout.addWidget(self.calc_splitting_checkbox, 3, 0, 1, 2)
        self.outlier_layout.addWidget(self.azim_q_override, 4, 0)
        self.outlier_layout.addWidget(self.azim_q, 4, 1)
        self.outlier_layout.addWidget(self.calc_spottiness_label, 5, 0)
        self.outlier_layout.addWidget(self.calc_spottiness_combobox, 5, 1)
        self.settings_layout.addWidget(self.outlier_settings, 4, 0, 6, 2)
        self.settings_layout.addWidget(self.defaults_button, 10, 0)

        self.setLayout(self.settings_layout)

    def toggle_outlier_settings(self):
        if self.calc_outlier_checkbox.isChecked():
            self.outlier_settings.show()
        else:
            self.outlier_settings.hide()
    
    def restore_defaults(self):
        self.madmult_override.setChecked(self.madmult_override_default)
        self.madmult.setValue(self.madmult_default)
        self.nbins_om_override.setChecked(self.nbins_om_override_default)
        self.nbins_om.setValue(self.nbins_om_default)
        self.calc_outlier_checkbox.setChecked(self.calc_outlier_default)
        self.calc_splitting_checkbox.setChecked(self.calc_splitting_default)
        self.calc_spottiness_combobox.setCurrentIndex(self.calc_spottiness_default)
        self.csim_first_spinbox.setValue(self.csim_first_default)


class main_window(QtWidgets.QWidget):
    """
    Main UI window.
    """
    def __init__(
            self,
            input_directory=None,
            output_directory=None,
            imctrl=None,
            flatfield=None,
            imgmask=None,
            bad_pixels=None,
            tth_integration_range=None,
            azim_integration_range=None,
            n_integration_bins=None,
            polarization=None,
            csim_first_index=None,
            outlier_mad_mult=None,
            n_mask_bins=None,
            azim_Q_ratio=None,
            outlier_option=None,
            spottiness_option=None,
            files_must_include=None,
            files_must_exclude=None,
        ):
        super().__init__()
        # Set up logging
        logging.getLogger('').setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s',datefmt='%m/%d/%Y %H:%M:%S')
        self.ch.setFormatter(self.formatter)
        logging.getLogger('').addHandler(self.ch)

        self.num_success = 0
        self.num_failed = 0

        self.input_directory_widget = FileSelectRowWidget(
            "Input Directory:",
            default_text=input_directory,
            isdir=True,
        )
        self.output_directory_widget = FileSelectRowWidget(
            "Output Directory:",
            default_text=output_directory,
            isdir=True,
        )
        # self.config_text = QtWidgets.QPushButton("Config file:")
        # self.config_loc = QtWidgets.QLabel()
        self.config_widget = ImctrlFileSelect(
            "Config file:",
            default_text=imctrl,
            startdir=self.input_directory_widget.file_name.text(),
            ext="Imctrl and PONI files (*.imctrl *.poni)",
        )
        self.config_widget.imctrl_set.connect(self.update_imctrl_data)
        # self.predef_mask_text = QtWidgets.QPushButton("Predefined Mask:")
        # self.predef_mask_loc = QtWidgets.QLabel()
        self.flatfield_widget = FileSelectRowWidget(
            "Flat-field file:",
            default_text=flatfield,
            startdir=self.input_directory_widget.file_name.text()
        )
        self.predef_mask_widget = FileSelectRowWidget(
            "Experimental Mask:",
            default_text=imgmask,
            startdir=self.input_directory_widget.file_name.text(),
            free_last_column=True,
        )
        self.bad_pixel_mask_widget = FileSelectRowWidget(
            "Bad Pixel Mask:",
            default_text=bad_pixels,
            startdir=self.input_directory_widget.file_name.text(),
        )

        self.poni_config_options = QtWidgets.QWidget()
        self.restore_default_config_options_button = QtWidgets.QPushButton("No config loaded")
        self.restore_default_config_options_button.setDisabled(True)
        self.restore_default_config_options_button.released.connect(self.update_imctrl_data)
        self.poni_default_text = QtWidgets.QLabel()
        self.poni_default_text.setWordWrap(True)
        self.iotth_label = QtWidgets.QLabel("2theta Integration Range:")
        self.iotth_min = QtWidgets.QDoubleSpinBox()
        self.iotth_max = QtWidgets.QDoubleSpinBox()
        self.azim_label = QtWidgets.QLabel("Azimuthal Integration Range:")
        self.azim_min = QtWidgets.QDoubleSpinBox()
        self.azim_min.setMaximum(360)
        self.azim_max = QtWidgets.QDoubleSpinBox()
        self.azim_max.setMaximum(360)
        self.outChannels_label = QtWidgets.QLabel("Number of Integration Bins:")
        self.outChannels = QtWidgets.QSpinBox()
        self.outChannels.setMaximum(100000)
        self.outChannels.setSingleStep(100)
        self.PolaVal_label = QtWidgets.QLabel("Polarization:")
        self.PolaVal = QtWidgets.QDoubleSpinBox()
        self.PolaVal.setMaximum(1.0)
        self.PolaVal.setSingleStep(0.1)
        self.poni_config_defaults = {
            self.iotth_min: 0.0,
            self.iotth_max: 10.0,
            self.azim_min: 0.0,
            self.azim_max: 360.0,
            self.outChannels: 2000,
            self.PolaVal: 1.0
        }
        for k, v in self.poni_config_defaults.items():
            k.setValue(v)

        self.advanced_settings_button = QtWidgets.QPushButton("Advanced Settings")
        self.advanced_settings_button.released.connect(self.advanced_settings_button_pressed)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.released.connect(self.start_button_pressed)
        self.clear_queue_button = QtWidgets.QPushButton("Clear Queue")
        self.clear_queue_button.released.connect(self.clear_queue_pressed)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.released.connect(self.stop_button_pressed)
        self.stop_button.setEnabled(False)
        # tooltip for clear queue button which notes that program will finish processing current item
        # exit button with "are you sure you're done processing everything" pop-up
        # self.process_existing_images_checkbox = QtWidgets.QCheckBox(
        #     "Process existing images"
        # )
        self.process_existing_only_radio = QtWidgets.QRadioButton("Process existing images only")
        self.process_both_radio = QtWidgets.QRadioButton("Process existing and new images")
        self.process_new_only_radio = QtWidgets.QRadioButton("Process new images only")
        self.process_both_radio.setChecked(True)
        self.regex_label = QtWidgets.QLabel("Regex for existing images:")
        self.existing_images_regex = QtWidgets.QTextEdit()

        self.process_existing_only_radio.released.connect(self.toggle_skip_processed_checkbox)
        self.process_both_radio.released.connect(self.toggle_skip_processed_checkbox)
        self.process_new_only_radio.released.connect(self.toggle_skip_processed_checkbox)
        self.skip_processed_files = QtWidgets.QCheckBox("Skip already-processed files")

        self.settings = {}
        self.settings_widget = AdvancedSettings(settings=self.settings)
        self.settings_shown = False

        # Update imctrl data if a file was passed, then update any passed override settings
        if imctrl is not None:
            self.update_imctrl_data()
        if tth_integration_range is not None:
            current = [self.iotth_min.value(), self.iotth_max.value()]
            try:
                self.iotth_min.setValue(tth_integration_range[0])
                self.iotth_max.setValue(tth_integration_range[1])
            except:
                self.iotth_min.setValue(current[0])
                self.iotth_max.setValue(current[1])
        if azim_integration_range is not None:
            current = [self.azim_min.value(), self.azim_max.value()]
            try:
                self.azim_min.setValue(azim_integration_range[0])
                self.azim_max.setValue(azim_integration_range[1])
            except:
                self.azim_min.setValue(current[0])
                self.azim_min.setValue(current[1])
        if n_integration_bins is not None:
            self.outChannels.setValue(n_integration_bins)
        if polarization is not None:
            self.PolaVal.setValue(polarization)
        if csim_first_index is not None:
            self.settings_widget.csim_first_spinbox.setValue(csim_first_index)
        if outlier_mad_mult is not None:
            self.settings_widget.madmult.setValue(outlier_mad_mult)
            self.settings_widget.madmult_override.setChecked(True)
        if n_mask_bins is not None:
            self.settings_widget.nbins_om.setValue(n_mask_bins)
            self.settings_widget.nbins_om_override.setChecked(True)
        if azim_Q_ratio is not None:
            self.settings_widget.azim_q.setValue(azim_Q_ratio)
            self.settings_widget.azim_q_override.setChecked(True)
        if outlier_option is not None:
            if outlier_option == "splitting":
                self.settings_widget.calc_outlier_checkbox.setChecked(True)
                self.settings_widget.calc_splitting_checkbox.setChecked(True)
            elif outlier_option == "outlier_only":
                self.settings_widget.calc_outlier_checkbox.setChecked(True)
                self.settings_widget.calc_splitting_checkbox.setChecked(False)
            elif outlier_option == "none":
                self.settings_widget.calc_outlier_checkbox.setChecked(False)
                self.settings_widget.calc_splitting_checkbox.setChecked(False)
        if spottiness_option is not None:
            if spottiness_option == "none":
                self.settings_widget.calc_spottiness_combobox.setCurrentIndex(0)
            elif spottiness_option == "spot_area_only":
                self.settings_widget.calc_spottiness_combobox.setCurrentIndex(1)
            elif spottiness_option == "spot_and_gradient":
                self.settings_widget.calc_spottiness_combobox.setCurrentIndex(2)
        if files_must_include is not None:
            self.settings_widget.regex_include_text.setText(files_must_include)
        if files_must_exclude is not None:
            self.settings_widget.regex_exclude_text.setText(files_must_exclude)

        self.cache_location = None

        # self.time_checkpoints = ["Start","Image loaded","Cache","Zero mask","Polar-correct","Outlier mask","Closing mask","Split first mask","Split second mask","All integrations","Save integrals","Delete project"]
        # self.time_checkpoints = ["Start", "Cache", "Zero mask", "Outlier mask", "Closing mask", "Splitting mask", "Integrations", "Save integrals", "CSim", "NMI", "SSim"]
        self.all_times = []
        # set up gsas-ii project
        # G2sc.blkSize = 2**8  # computer-dependent tuning parameter
        self.blkSize = 2**8
        # G2sc.SetPrintLevel('warn')   # reduces output

        self.cache = {}  # place to save intermediate computations

        self.queue = deque()
        # self.event_handler = image_monitor(self.queue)
        self.stop_event = threading.Event()
        # self.watchdog_thread = threading.Thread(target=watchdog_observer,args=(self.directory,self.event_handler),daemon=True)

        self.timer = QtCore.QTimer()
        self.keep_running = False
        self.timer.timeout.connect(self.on_timeout)

        # self.iteration_thread = QtCore.QThread()
        # self.cache_thread = QtCore.QThread()

        self.queue_length_info = QtWidgets.QLabel(
            f"Queue is {len(self.queue)} items long"
        )
        self.num_success_info = QtWidgets.QLabel(
            f"Files completed: {self.num_success}"
        )
        self.num_failed_info = QtWidgets.QLabel(
            f"Errored files: {self.num_failed}"
        )

        self.list_of_times = []
        self.list_of_time_names = []

        self.open_resultsUI_button = QtWidgets.QPushButton("Open Data Viewer")
        self.open_resultsUI_button.released.connect(self.open_resultsUI)
        self.open_maskwidget_button = QtWidgets.QPushButton("Open Mask Creation Program")
        self.open_maskwidget_button.released.connect(self.open_maskwidget)

        # self.is_running_process = False

        self.poni_config_options_layout = QtWidgets.QGridLayout()
        self.poni_config_options.setLayout(self.poni_config_options_layout)
        self.poni_config_options_layout.addWidget(self.restore_default_config_options_button, 0, 0)
        self.poni_config_options_layout.addWidget(self.poni_default_text, 1, 0, 3, 1)
        self.poni_config_options_layout.addWidget(self.iotth_label, 0, 1)
        self.poni_config_options_layout.addWidget(self.iotth_min, 0, 2)
        self.poni_config_options_layout.addWidget(self.iotth_max, 0, 3)
        self.poni_config_options_layout.addWidget(self.azim_label, 1, 1)
        self.poni_config_options_layout.addWidget(self.azim_min, 1, 2)
        self.poni_config_options_layout.addWidget(self.azim_max, 1, 3)
        self.poni_config_options_layout.addWidget(self.outChannels_label, 2, 1)
        self.poni_config_options_layout.addWidget(self.outChannels, 2, 2)
        self.poni_config_options_layout.addWidget(self.PolaVal_label, 3, 1)
        self.poni_config_options_layout.addWidget(self.PolaVal, 3, 2)

        self.window_layout = QtWidgets.QGridLayout()
        self.window_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.window_layout.addWidget(self.input_directory_widget, 0, 0, 1, 3)
        self.window_layout.addWidget(self.output_directory_widget, 1, 0, 1, 3)
        self.window_layout.addWidget(self.config_widget, 2, 0, 1, 3)
        self.window_layout.addWidget(self.poni_config_options, 3, 1, 3, 2)
        self.window_layout.addWidget(self.flatfield_widget, 6, 0, 1, 3)
        self.window_layout.addWidget(self.predef_mask_widget, 7, 0, 1, 3)
        self.predef_mask_widget.layout().addWidget(self.open_maskwidget_button, 0, 3)
        self.window_layout.addWidget(self.bad_pixel_mask_widget, 8, 0, 1, 3)
        self.window_layout.addWidget(self.advanced_settings_button, 10, 0)
        self.window_layout.addWidget(self.start_button, 9, 0)
        self.window_layout.addWidget(self.clear_queue_button, 9, 1)
        self.window_layout.addWidget(self.stop_button, 9, 2)
        self.window_layout.addWidget(self.settings_widget, 11, 0, 1, 3)
        self.window_layout.addWidget(self.process_existing_only_radio, 12, 0)
        self.window_layout.addWidget(self.process_both_radio, 12, 1)
        self.window_layout.addWidget(self.process_new_only_radio, 12, 2)
        self.window_layout.addWidget(self.skip_processed_files, 13, 0)
        self.window_layout.addWidget(self.queue_length_info, 14, 0)
        self.window_layout.addWidget(self.num_success_info, 14, 1)
        self.window_layout.addWidget(self.num_failed_info, 14, 2)
        self.window_layout.addWidget(self.open_resultsUI_button, 15, 2)
        # self.window_layout.addWidget(self.regex_label,7,0)
        # self.window_layout.addWidget(self.existing_images_regex,8,0)
        self.settings_widget.hide()

        self.setLayout(self.window_layout)
        self.show()

    def update_imctrl_data(self):
        """
        Load in some modifiable image controls. Called when an image control
        file is selected.
        """
        self.restore_default_config_options_button.setEnabled(True)
        local_controls = {}
        imctrl = self.config_widget.file_name.text()
        ext = os.path.splitext(imctrl)[1]
        if ext == ".imctrl":
            self.restore_default_config_options_button.setText("Restore Config Values")
            self.poni_default_text.setText("")
            with open(imctrl, "r") as imctrlfile:
                lines = imctrlfile.readlines()
                LoadControls(lines, local_controls)
            self.iotth_min.setValue(local_controls["IOtth"][0])
            self.iotth_max.setValue(local_controls["IOtth"][1])
            self.azim_min.setValue(local_controls["LRazimuth"][0])
            self.azim_max.setValue(local_controls["LRazimuth"][1])
            self.outChannels.setValue(local_controls["outChannels"])
            self.PolaVal.setValue(local_controls["PolaVal"][0])
        # reset to 0 if swapping to poni
        # may want ability to set values before loading in
        elif ext == ".poni":
            self.restore_default_config_options_button.setText("Restore Defaults")
            self.poni_default_text.setText("Poni files do not contain this information. Please adjust the defaults as appropriate.")
            for k, v in self.poni_config_defaults.items():
                k.setValue(v)
        elif imctrl == "":
            self.restore_default_config_options_button.setText("No config loaded")
            self.restore_default_config_options_button.setDisabled(True)
            for k, v in self.poni_config_defaults.items():
                k.setValue(v)

    def set_cache_location(self, cache_location):
        self.cache_location = cache_location

    def cache_thread_finished(self):
        self.has_made_cache = True
        self.cache_thread = None
        if self.keep_running:
            self.timer.start()

    def iteration_thread_finished(self):
        if self.keep_running:
            self.timer.start()

    def on_timeout(self):
        """
        Called regularly while the pipeline is running. Whenever there is a new
        image in the queue, this will start up a thread for the cache (if not yet
        run) or for processing the new image. Removes the processed image from
        the queue.
        """
        if self.keep_running:
            # block=True in Queue.get() tells it to wait until there is something in the queue to grab it
            # can also set a timeout value (in seconds) to wait before it throws an Empty exception
            # Windows systems apparently have a problem with block=True, timeout=None
            # filename,name,number = queue.get(block=True,timeout=30)
            if self.queue:
                self.queue_length_info.setText(
                    f"Queue is {len(self.queue)} items long"
                )
                self.num_success_info.setText(
                    f"Files completed: {self.num_success}"
                )
                self.num_failed_info.setText(
                    f"Errored files: {self.num_failed}"
                )
                if self.has_made_cache and self.cache_location is not None:
                    # ensure it's been some time since the file was modified
                    if time.time() - os.path.getmtime(self.queue[0][0]) > 1:
                        filename, name, number, ext = self.queue.popleft()
                        print(filename, name, number, ext)
                        # print("Queue is {0} items long".format(len(self.queue)))
                        # self.single_iteration(filename,self.imgctrl,self.imgmask,self.directory,name,number)
                        # set up iteration thread. Should set these up with a pool and just run, but for now, run one at a time.
                        self.timer.stop()
                        self.iteration_thread = QtCore.QThread()
                        if not self.settings_widget.azim_q_override.isChecked():
                            self.iteration_worker = SingleIterator(
                                filename,
                                self.imgctrl,
                                self.imgmask,
                                self.input_directory,
                                self.output_directory,
                                name,
                                number,
                                ext,
                                cache_location = self.cache_location,
                                calc_outlier = self.settings_widget.calc_outlier_checkbox.isChecked(),
                                calc_splitting = self.settings_widget.calc_splitting_checkbox.isChecked(),
                                calc_spot_stats = self.settings_widget.calc_spottiness_combobox.currentIndex() != 0,
                                calc_grad_spottiness = self.settings_widget.calc_spottiness_combobox.currentIndex() == 2,
                                csim_first_index = self.settings_widget.csim_first_spinbox.value(),
                                timing = self.list_of_times,
                                timing_names = self.list_of_time_names,
                            )
                        else:
                            self.iteration_worker = SingleIterator(
                                filename,
                                self.imgctrl,
                                self.imgmask,
                                self.input_directory,
                                self.output_directory,
                                name,
                                number,
                                ext,
                                cache_location = self.cache_location,
                                azim_Q_shape_min = self.settings_widget.azim_q.value(),
                                calc_outlier = self.settings_widget.calc_outlier_checkbox.isChecked(),
                                calc_splitting = self.settings_widget.calc_splitting_checkbox.isChecked(),
                                calc_spot_stats = self.settings_widget.calc_spottiness_combobox.currentIndex() != 0,
                                calc_grad_spottiness = self.settings_widget.calc_spottiness_combobox.currentIndex() == 2,
                                csim_first_index = self.settings_widget.csim_first_spinbox.value(),
                                timing = self.list_of_times,
                                timing_names = self.list_of_time_names,
                            )
                        self.iteration_worker.moveToThread(self.iteration_thread)
                        self.iteration_thread.started.connect(self.iteration_worker.run)
                        self.iteration_worker.finished.connect(
                            self.iteration_thread.quit
                        )
                        self.iteration_worker.finished.connect(
                            self.iteration_worker.deleteLater
                        )
                        self.iteration_worker.succeeded.connect(
                            self.increment_successful_completion
                        )
                        self.iteration_worker.failed.connect(
                            self.increment_failed_completion
                        )
                        # self.iteration_thread.finished.connect(self.iteration_thread.deleteLater)
                        self.iteration_thread.finished.connect(
                            self.iteration_thread_finished
                        )
                        self.iteration_thread.start()
                        # print("Queue is {0} items long".format(len(self.queue)))
                        # self.queue_length_info.setText("Queue is {0} items long".format(len(self.queue)))
                        # test_iteration(filename)
                        # wait_start = time.time()
                else:
                    # check that it's been a moment
                    if time.time() - os.path.getmtime(self.queue[0][0]) > 1:
                        # set up cache thread
                        self.timer.stop()
                        self.cache_thread = QtCore.QThread()
                        filename = self.queue[0][0]
                        # print(filename)
                        esdMul = self.settings_widget.madmult_default
                        if self.settings_widget.madmult_override.isChecked():
                            esdMul = self.settings_widget.madmult.value()
                        if self.iotth_max.value() != 0.0:
                            tth_integration_range = [
                                self.iotth_min.value(),
                                self.iotth_max.value()
                            ]
                        else:
                            tth_integration_range = None
                        if (self.azim_min.value() != 0.0) or (os.path.splitext(self.imgctrl)[1] == ".poni"):
                            azim_integration_range = [
                                self.azim_min.value(),
                                self.azim_max.value()
                            ]
                        else:
                            azim_integration_range = None
                        if self.outChannels.value() != 0.0:
                            n_integration_bins = self.outChannels.value()
                        else:
                            n_integration_bins = None
                        if self.PolaVal.value() != 0.0:
                            polarization = [self.PolaVal.value(), False]
                        else:
                            polarization = None
                        self.cache_worker = CacheCreator(
                            self.cache,
                            self.input_directory,
                            self.output_directory,
                            filename,
                            self.imgctrl,
                            self.flatfield,
                            self.imgmask,
                            self.bad_pixels,
                            self.blkSize,
                            esdMul = esdMul,
                            tth_integration_range=tth_integration_range,
                            azim_integration_range=azim_integration_range,
                            n_integration_bins=n_integration_bins,
                            polarization=polarization,
                        )
                        self.cache_worker.moveToThread(self.cache_thread)
                        self.cache_thread.started.connect(self.cache_worker.run)
                        self.cache_worker.finished.connect(self.cache_thread.quit)
                        self.cache_worker.finished.connect(
                            self.cache_worker.deleteLater
                        )
                        self.cache_worker.cache_location_signal.connect(self.set_cache_location)
                        # self.cache_thread.finished.connect(self.cache_thread.deleteLater)
                        self.cache_thread.finished.connect(self.cache_thread_finished)
                        self.cache_thread.start()
            else:
                self.queue_length_info.setText("Queue is 0 items long")
                self.num_success_info.setText(
                    f"Files completed: {self.num_success}"
                )
                self.num_failed_info.setText(
                    f"Errored files: {self.num_failed}"
                )
                if self.process_existing_only_radio.isChecked():
                    self.stop_button_pressed()
            # else:
            #    #If it's been over an hour since the last update, stop
            #    if time.time() - wait_start > 60:
            #        print("Average time: {0}".format(np.average(self.all_times)))
            #        self.keep_running = False
        else:
            self.timer.stop()

    def start_processing(self):
        """
        Called by the start button. Gathers the provided information,
        creates the queue, starts the image directory monitor, and starts the timer
        which runs on_timeout() periodically.
        """
        self.input_directory = self.input_directory_widget.file_name.text()
        self.output_directory = self.output_directory_widget.file_name.text()
        self.imgctrl = self.config_widget.file_name.text()
        self.imgmask = self.predef_mask_widget.file_name.text()
        self.flatfield = self.flatfield_widget.file_name.text()
        self.bad_pixels = self.bad_pixel_mask_widget.file_name.text()
        self.include_regex = self.settings_widget.regex_include_text.text()
        self.exclude_regex = self.settings_widget.regex_exclude_text.text()
        self.cache = {}
        self.has_made_cache = False
        # print("Directory: {0}, Ctrl file: {1}, Predef mask: {2}".format(dir_name,ctrl_name,predef_mask))
        # self.process = main_process(dir_name,ctrl_name,predef_mask)
        # create subdirectories if needed
        newdirs = ["maps", "masks", "integrals", "stats", "logs"]
        if not ((self.flatfield is None) or (self.flatfield == "")):
            newdirs.append("flatfield")
        for newdir in newdirs:
            path = os.path.join(self.output_directory, newdir)  # store maps with the images
            if not os.path.exists(path):
                os.mkdir(path)

        # Set up logging
        curtime = time.strftime('%Y_%m_%d_%H_%M_%S')
        self.logging_filepath = os.path.join(self.output_directory, 'logs', f'{curtime}.log')
        self.fh = logging.FileHandler(self.logging_filepath)
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(self.formatter)
        logging.getLogger('').addHandler(self.fh)

        self.num_success = 0
        self.num_failed = 0

        # Grab existing file names and add them to the queue if option checked
        if self.process_both_radio.isChecked() or self.process_existing_only_radio.isChecked():
            existing_files = sorted(
                glob.glob(self.input_directory + "/*.tif"),
                # ctime is not platform-independent, so using mtime
                key = os.path.getmtime
            )
            reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
            if (self.include_regex is not None) and (self.include_regex.strip() != ""):
                reg_include = r"(?P<input_directory>.*[\\\/])(?P<name>.*" + re.escape(self.include_regex) + r".*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
                regs = reg_include
            else:
                regs = reg_image
            ignore_regs = None
            if (self.exclude_regex is not None) and (self.exclude_regex.strip() != ""):
                ignore_regs = r".*" + re.escape(self.exclude_regex) + r".*"
            for filename in existing_files:
                results = re.match(regs, filename)
                if results is not None and ignore_regs is not None:
                    if re.match(ignore_regs, filename):
                        continue
                # If skipping already-processed files, check for an existing integral file for that name+number
                if self.skip_processed_files.isChecked():
                    integral_filename = os.path.join(
                        self.output_directory,
                        "integrals",
                        results.group("name") + "-" + results.group("number") + "_base.chi"
                    )
                    if os.path.exists(integral_filename):
                        logging.getLogger('').info(f"Skipping {filename}")
                        continue
                if results is not None:
                    self.queue.append(
                        [
                            filename,
                            results.group("name"),
                            results.group("number"),
                            results.group("ext"),
                        ]
                    )
                    print(
                        filename,
                        results.group("name"),
                        results.group("number"),
                        results.group("ext"),
                    )

        # Start queue
        print("Starting queue")

        if self.process_new_only_radio.isChecked() or self.process_both_radio.isChecked():
            self.observer = Observer()
            self.event_handler = ImageMonitor(self.queue, include = self.include_regex, exclude = self.exclude_regex)
            self.observer.schedule(self.event_handler, self.input_directory, recursive=False)
            self.observer.start()

        # main function to cycle, calls iteration while there are new images to process
        self.keep_running = True
        self.csim_first = []
        self.csim_prev = []
        self.nmi_first = []
        self.nmi_prev = []
        self.ssim_first = []
        self.ssim_prev = []
        self.timer.start(100)

    def clear_queue(self):
        self.queue.clear()
        self.queue_length_info.setText(
            f"Queue is {len(self.queue)} items long"
        )
        self.num_success_info.setText(
            f"Files completed: {self.num_success}"
        )
        self.num_failed_info.setText(
            f"Errored files: {self.num_failed}"
        )

    def pause(self):
        """
        Prevents new images from being processed. Any image currently being processed
        will continue. The image directory monitor will continue to run and populate
        the queue.
        Called when the Pause button is pressed.
        """
        print("Pausing. If processing an image, that process will complete first.")
        self.keep_running = False
        # watchdog thread will still keep populating the queue

    def resume(self):
        """
        Resume after pausing.
        """
        self.keep_running = True
        self.timer.start(100)

    def advanced_settings_button_pressed(self):
        """
        Show/hide the advanced settings widget.
        """
        if self.settings_shown:
            self.settings_shown = False
            self.settings_widget.hide()
        else:
            self.settings_shown = True
            self.settings_widget.show()

    def start_button_pressed(self):
        if self.start_button.text() == "Start":
            if os.path.splitext(self.config_widget.file_name.text())[1] == ".poni":
                if ((self.iotth_max.value() == 0) or (self.outChannels.value() == 0) or (self.PolaVal.value() == 0)):
                    print("Please specify the 2theta and azimuthal integration range, number of integration bins, and polarization value.")
                    return
            self.start_processing()
            self.start_button.setText("Pause")
            self.stop_button.setEnabled(True)
            self.input_directory_widget.setEnabled(False)
            self.output_directory_widget.setEnabled(False)
            self.config_widget.setEnabled(False)
            self.flatfield_widget.setEnabled(False)
            self.predef_mask_widget.setEnabled(False)
            self.bad_pixel_mask_widget.setEnabled(False)
            self.process_existing_only_radio.setEnabled(False)
            self.process_both_radio.setEnabled(False)
            self.process_new_only_radio.setEnabled(False)
            self.poni_config_options.setEnabled(False)
        elif self.start_button.text() == "Pause":
            self.pause()
            self.start_button.setText("Resume")
        elif self.start_button.text() == "Resume":
            self.resume()
            self.start_button.setText("Pause")

    def clear_queue_pressed(self):
        print("Clearing queue")
        self.clear_queue()

    def stop_button_pressed(self):
        """
        Stop processing new images, clear the queue, and gather and
        display any timing information. If the cache is running, sends
        a signal to tell it to stop at the next checkpoint.
        If a processing thread is still running, this also connects the
        thread closing signal to the really_stopped() function which
        resets the UI. Otherwise, this function is called directly.
        """
        print("Stopping and clearing queue")
        # print(f"Length of timing list: {len(self.list_of_times)}")
        # print(f"Mean time: {np.mean(self.list_of_times):.4f} +/- {np.std(self.list_of_times):.4f}")
        if len(self.list_of_times) > 0:
            try:
                means = np.mean(self.list_of_times, axis=0)
                std = np.std(self.list_of_times, axis=0)
                logging.getLogger('').info(f"Finished successfully processing {self.num_success} files. {self.num_failed} files encountered an error.")
                formatter = logging.Formatter('%(message)s')
                self.fh.setFormatter(formatter)
                self.ch.setFormatter(formatter)
                for i in range(len(self.list_of_time_names)):
                    logging.getLogger('').info(f"{self.list_of_time_names[i]}: {means[i]:.4f} +/- {std[i]:.4f}")
                self.fh.setFormatter(self.formatter)
                self.ch.setFormatter(self.formatter)
            except:
                self.fh.setFormatter(self.formatter)
                logging.getLogger('').warning("Problem printing out timing info")
        self.list_of_times = []
        self.list_of_time_names = []
        self.stop_button.setText("Stopping...")
        # disable all
        self.advanced_settings_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.clear_queue_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        # self.pause()
        self.keep_running = False
        self.clear_queue()
        # self.watchdog_thread.stop()
        # self.watchdog_thread.join()
        if self.process_new_only_radio.isChecked() or self.process_both_radio.isChecked():
            self.observer.stop()
            self.observer.join()
        # if self.cache_thread.isRunning():
        if not self.has_made_cache:
            self.cache_worker.stopEarly = True
            self.cache_thread.quit()
            self.cache_thread.finished.connect(self.really_stopped)
            self.cache_thread.finished.connect(self.cache_thread.deleteLater)
        elif self.iteration_thread.isRunning():
            self.iteration_thread.quit()
            self.iteration_thread.finished.connect(self.really_stopped)
            self.iteration_thread.finished.connect(self.iteration_thread.deleteLater)
        else:
            # self.is_running_process = False
            self.really_stopped()

    def really_stopped(self):
        """
        Function called when processing has been fully stopped.
        Resets the UI to be interactable again.
        Also checks if the log file is empty; if so, it is deleted.
        """
        # self.is_running_process = False
        print("Stopped")
        self.advanced_settings_button.setEnabled(True)
        self.start_button.setText("Start")
        self.start_button.setEnabled(True)
        self.clear_queue_button.setEnabled(True)
        self.stop_button.setText("Stop")
        self.input_directory_widget.setEnabled(True)
        self.output_directory_widget.setEnabled(True)
        self.config_widget.setEnabled(True)
        self.flatfield_widget.setEnabled(True)
        self.predef_mask_widget.setEnabled(True)
        self.bad_pixel_mask_widget.setEnabled(True)
        self.process_existing_only_radio.setEnabled(True)
        self.process_both_radio.setEnabled(True)
        self.process_new_only_radio.setEnabled(True)
        self.poni_config_options.setEnabled(True)

        # Remove file handler
        if len(logging.getLogger('').handlers) > 1:
            logging.getLogger('').removeHandler(logging.getLogger('').handlers[1])

    def open_maskwidget(self):
        """
        Open the mask widget in a separate window and output its result back to the pipeline.
        """
        self.mask_widget = MainWindow(opened_from_pipeline=True, imctrl_file=self.config_widget.file_name.text())
        self.mask_widget.show()
        self.mask_widget.mask_location.connect(self.update_predef_mask)

    def update_predef_mask(self, location):
        self.predef_mask_widget.file_name.setText(location)

    def toggle_skip_processed_checkbox(self):
        if self.process_existing_only_radio.isChecked() or self.process_both_radio.isChecked():
            self.skip_processed_files.setEnabled(True)
        elif self.process_new_only_radio.isChecked():
            self.skip_processed_files.setEnabled(False)

    def increment_successful_completion(self):
        self.num_success += 1

    def increment_failed_completion(self):
        self.num_failed += 1

    def open_resultsUI(self):
        """
        Open the results UI in a separate window.
        Launches a new terminal which starts the current virtual environment and runs the results UI there
        to avoid slowing down the processing speed for both the pipeline and the UI.
        """
        # Read in input directory, output directory, and config file information
        # Use local values to not interfere with the pipeline just in case
        input_directory = self.input_directory_widget.file_name.text()
        output_directory = self.output_directory_widget.file_name.text()
        imgctrl = self.config_widget.file_name.text()
        # Overwrite any modified settings
        # UI checks for number of integration bins and wavelength
        outChannels = self.outChannels.value()
        # Pass along info
        args = ""
        args_list = []
        if input_directory != "":
            args += f" -i \"{input_directory}\""
            args_list.append("-i")
            args_list.append(input_directory)
        if output_directory != "":
            args += f" -o \"{output_directory}\""
            args_list.append("-o")
            args_list.append(output_directory)
        if imgctrl != "":
            args += f" -c \"{imgctrl}\""
            args_list.append("-c")
            args_list.append(imgctrl)
        # If the pipeline is currently running, skip the directory prompt entirely
        if not self.input_directory_widget.isEnabled():
            args += f" -s"
            args_list.append("-s")
            # pass along input data to the settings
            if outChannels != 0.0:
                args += f" -r -b {outChannels}"
                args_list.append("-r")
                args_list.append("-b")
                args_list.append(str(outChannels))
        # Otherwise simply auto-fill the directory prompt (handled with arguments)
        # Launch the results UI
        current_venv_exe = sys.executable
        venv_path = os.path.dirname(current_venv_exe)
        directory = os.path.dirname(os.path.realpath(__file__))
        results_UI_location = os.path.join(directory,"pyqtgraph_layout.py")
        if sys.platform.startswith("win"):
            activate_cmd = os.path.join(venv_path, "activate.bat")
            command = f"start cmd.exe /k \"{activate_cmd} && {current_venv_exe} {results_UI_location} {args}\""
            subprocess.Popen(command, shell=True)
        elif sys.platform == "linux":
            # Check if this is a conda env
            if os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
                # find current environment name
                conda_env = os.environ['CONDA_PREFIX']
                # terminal_command = f"conda activate {conda_env} && python {results_UI_location} {args}"
                terminal_command = ["conda","run","-p",conda_env,"python",results_UI_location,*args_list]
            else:
                activate_cmd = os.path.join(venv_path, "activate")
                terminal_command = f"source {activate_cmd} && python {results_UI_location} {args}"
            # Find which terminal emulator is installed
            for term in ("lxterminal", "gnome-terminal", "konsole", "xterm",
                         "terminator", "terminology", "tilix"):
                try:
                    found_terminal = shutil.which(term)
                    if not found: continue
                except AttributeError:
                    logging.getLogger('').exception(f"Error running shutil.which({term}). Skipping.")
                if term == "xterm":
                    # subprocess_command = f"xterm -e {terminal_command}"
                    subprocess_command = ["xterm","-e"]
                    # subprocess_command = f"xterm -hold -e {terminal_command}"
                    break
                elif term == "gnome-terminal":
                    subprocess_command = f"gnome-terminal {terminal_command}"
                    break
                # rest not yet tested
                elif term == "lxterminal":
                    subprocess_command = f"lxterminal -e {terminal_command}"
                    break
                elif term == "terminator":
                    subprocess_command = f"terminator -x {terminal_command}"
                    break
                elif term == "konsole":
                    subprocess_command = f"konsole -p --hold -e {terminal_command}"
                    break
                elif term == "tilix":
                    subprocess_command = f"tilix -e {terminal_command}"
                    break
                elif term == "terminology":
                    subprocess_command = f"terminology --hold -e {terminal_command}"
                    break
                else:
                    logging.getLogger('').warning(f"No terminal emulator found for Linux environment. Cannot open new terminal.")
                    return
            try:
                if term == "xterm":
                    subprocess.Popen([*subprocess_command, *terminal_command], start_new_session=True)
                else:
                    subprocess.Popen(subprocess_command, shell=True)
            except:
                logging.getLogger('').exception(f"Problem launching new subprocess terminal with command {subprocess_command}.")
        else:
            print("Platform not yet supported.")

    def closeEvent(self, evt):
        """
        Function called when hitting the X button in the corner to close
        the window. Interrupts the action with a prompt if images are still
        being processed.
        """
        # if not self.is_running_process:
        #    evt.accept()
        if not self.stop_button.isEnabled():
            # button disabled while not actively running a process; no need to prompt
            evt.accept()
        elif (
            QtWidgets.QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to stop running over all data?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            == QtWidgets.QMessageBox.StandardButton.Yes
        ):
            self.clear_queue()
            self.pause()
            evt.accept()
        else:
            evt.ignore()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="Location of the input image directory")
    parser.add_argument("-o", "--output_directory", help="Location to place the output files from this pipeline")
    parser.add_argument("-c", "--imctrl", help="Image control file")
    parser.add_argument("-f", "--flatfield", help="Flatfield file")
    parser.add_argument("-m", "--imgmask", help="Experimental mask")
    parser.add_argument("-b", "--bad_pixels", help="Detector known bad pixel mask")
    parser.add_argument("-t", "--tth_integration_range", nargs=2, type=float, help= "2theta integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("-z", "--azim_integration_range", nargs=2, type=float, help="Azimuthal integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("--n_integration_bins", type=float, help="Number of bins to use for integration, if overriding or not included in the config file.")
    parser.add_argument("-p", "--polarization", type=float, help="Polarization of the image")
    parser.add_argument("--csim_first_index", help="Numerical index for the file which should be considered first when calculating cosine similarity.")
    # parser.add_argument("--outlier_mad_mult", type=float, default=3, help="Multiplier of median absolute deviation to use when considering a value an outlier.")
    parser.add_argument("--outlier_mad_mult", type=float, help="Multiplier of median absolute deviation to use when considering a value an outlier. Default is 3.")
    parser.add_argument("--n_mask_bins", type=int, help="Number of bins used when calculating outliers.")
    # parser.add_argument("-a", "--azim_Q_ratio", type=int, default=100, help="Azimuthal to Q width ratio used for classifying spots.")
    parser.add_argument("-a", "--azim_Q_ratio", type=int, help="Azimuthal to Q width ratio used for classifying spots. Default is 100.")
    parser.add_argument("--outlier_option", choices=["splitting", "outlier_only", "none"], default="splitting", help="Choose whether to perform no outlier masking, outlier masking only, or outlier masking with spot/texture splitting.")
    parser.add_argument("--spottiness_option", choices=["spot_and_gradient","spot_area_only","none"], default="spot_area_only", help="Choose whether to perform spottiness statistics calculations.")
    parser.add_argument("--files_must_include", help="Process only files in the directory which include the provided string in their name.")
    parser.add_argument("--files_must_exclude", help="Exclude files in the directory which have the provided string in their name.")
    # parser.add_argument("-n", "--no_ui", action="store_true", help="Skip the UI and run over files with the specified options (otherwise options will be pre-filled in the UI). Must include input directory, output directory, and image control file to run.")
    args = parser.parse_args()

    # Pass in location and names of files
    dataLoc = os.path.abspath(
        os.path.split(__file__)[0]
    )  # data in location of this file
    PathWrap = lambda fil: os.path.join(
        dataLoc, fil
    )  # convenience function for file paths

    if args.flatfield is not None:
        flatfield = PathWrap(args.flatfield)
    else:
        flatfield = None
    if args.imgmask is not None:
        imgmask = PathWrap(args.imgmask)
    else:
        imgmask = None
    if args.bad_pixels is not None:
        bad_pixels = PathWrap(args.bad_pixels)
    else:
        bad_pixels = None
    if args.input_directory:
        input_directory = PathWrap(args.input_directory)
    else:
        input_directory = None
    if args.output_directory:
        output_directory = PathWrap(args.output_directory)
    else:
        output_directory = None
    if args.imctrl:
        if os.path.exists(PathWrap(args.imctrl)):
            imgctrl = PathWrap(args.imctrl)
        elif os.path.exists(os.path.join(input_directory, args.imctrl)):
            imgctrl = os.path.join(input_directory, args.imctrl)
        else:
            print(
                "Image control file not found in this directory or in specified directory."
            )
            imgctrl = None
    else:
        imgctrl = None

    app = QtWidgets.QApplication([])
    window = main_window(
        input_directory=input_directory,
        output_directory=output_directory,
        imctrl=imgctrl,
        flatfield=flatfield,
        imgmask=imgmask,
        bad_pixels=bad_pixels,
        tth_integration_range=args.tth_integration_range,
        azim_integration_range=args.azim_integration_range,
        n_integration_bins=args.n_integration_bins,
        polarization=args.polarization,
        csim_first_index=args.csim_first_index,
        outlier_mad_mult=args.outlier_mad_mult,
        n_mask_bins=args.n_mask_bins,
        azim_Q_ratio=args.azim_Q_ratio,
        outlier_option=args.outlier_option,
        spottiness_option=args.spottiness_option,
        files_must_include=args.files_must_include,
        files_must_exclude=args.files_must_exclude,
    )
    sys.exit(app.exec())
