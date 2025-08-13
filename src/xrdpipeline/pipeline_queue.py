from collections import deque
import argparse
import copy
import glob
import re
import os, sys
import time
import threading

from PIL import Image

import PySide6
from pyqtgraph.Qt import QtCore, QtWidgets

from scipy import spatial

from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer

from GSASII_imports import *
from pipeline import run_iteration
from cache_creation import getmaps, get_azimbands, prepare_qmaps, gradient_cache
from corrections_and_maps import get_Qbands

class image_monitor(RegexMatchingEventHandler):
    def __init__(self, queue):
        # dir\name_number_ext.tif or dir\name-number_ext.tif
        #'number' may be 00000 or xxxxx_xxxxx or xxxxx-xxxxx
        #'_ext' not on base images
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif.metadata$"
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif$"
        reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
        # reg for integral data files

        ###
        regs = [reg_image]
        RegexMatchingEventHandler.__init__(self, regexes=regs)
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


class file_select(QtWidgets.QWidget):
    def __init__(self, label, default_text=None, isdir=False, startdir=".", ext=None):
        super().__init__()
        self.setMinimumWidth(600)
        # self.label = QtWidgets.QLabel(label)
        self.file_select_button = QtWidgets.QPushButton(label)
        # self.file_select_button = QtWidgets.QPushButton(self.default_text)
        self.file_name = QtWidgets.QLabel(default_text)
        self.isdir = isdir
        self.startdir = startdir
        self.ext = ext

        self.layout = QtWidgets.QGridLayout()
        # self.layout.addWidget(self.label,0,0)
        # self.layout.addWidget(self.file_select_button,0,1,1,3)
        self.layout.addWidget(self.file_select_button, 0, 0)
        self.layout.addWidget(self.file_name, 0, 1, 1, 3)
        self.setLayout(self.layout)

        self.file_select_button.released.connect(self.select_file)

    def select_file(self):
        if self.isdir:
            location = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Select Directory"
            )
            # self.file_select_button.setText(location)
            self.file_name.setText(location)
        else:
            location = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select File", self.startdir, self.ext
            )
            # self.file_select_button.setText(location[0])
            self.file_name.setText(location[0])


class imctrl_file_select(file_select):
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
        calc_outlier = True,
        esdMul = 3.0,
        outChannels = None,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        calc_spottiness = False,
        not_in_poni_settings = {},
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
        self.calc_outlier = calc_outlier
        self.esdMul = esdMul
        self.outChannels = outChannels
        self.calc_splitting = calc_splitting
        self.azim_Q_shape_min = azim_Q_shape_min
        self.calc_spottiness = calc_spottiness
        self.not_in_poni_settings = not_in_poni_settings
        self.stopEarly = False

    def run(self):
        cache_time = time.time()
        if self.logging:
            print("Creating cache")
        image_dict = read_image(self.filename)
        # img.loadControls(imctrlname)   # set controls/calibrations/masks
        if os.path.splitext(self.imctrlname)[1] == ".imctrl":
            with open(self.imctrlname, "r") as imctrlfile:
                lines = imctrlfile.readlines()
                LoadControls(lines, image_dict["Image Controls"])
        else:
            with open(self.imctrlname, "r") as imctrlfile:
                lines = imctrlfile.readlines()
                LoadControlsPONI(lines, image_dict["Image Controls"])
        for k, v in self.not_in_poni_settings.items():
            image_dict["Image Controls"][k] = v
        # cache['Image Controls'] = img.getControls() # save controls & masks contents for quick reload
        # self.cache['image'] = tf.imread(self.filename)
        self.cache["image"] = load_image(self.filename)
        if self.stopEarly: return

        predef_mask = {}
        if (self.imgmaskname is not None) and (self.imgmaskname != ""):
            # img.loadMasks(imgmaskname)
            suffix = self.imgmaskname.split(".")[1]
            if suffix == "immask":
                readMasks(self.imgmaskname, image_dict["Masks"], False)
            elif suffix == "tif":
                print(self.imgmaskname)
                predef_mask = read_image(self.imgmaskname)
        else:
            predef_mask["image"] = np.zeros_like(image_dict["image"], dtype=bool)
        if (self.bad_pixels is not None) and (self.bad_pixels != ""):
            suffix = self.bad_pixels.split(".")[1]
            if suffix == "tif":
                bad_pixel_mask = read_image(self.bad_pixels)
                predef_mask |= bad_pixel_mask
            else:
                print("Unsupported bad pixel mask image type. Skipping file read. Any zero-intensity pixels will automatically be masked.")
        self.cache["predef_mask"] = predef_mask

        flatfield_image = None
        if (self.flatfield is not None) and (self.flatfield != ""):
            # flatfield_image = tf.imread(self.flatfield)
            flatfield_image = load_image(self.flatfield)
        self.cache["flatfield"] = flatfield_image
        if self.stopEarly: return
        
        # imctrlname = imctrlname.split("\\")[-1].split('/')[-1]
        # path1 =  os.path.join(pathmaps,imctrlname)
        # im = Image.fromarray(TA[0])
        # im.save(os.path.splitext(path1)[0] + '_2thetamap.tif')
        imsave = Image.fromarray(predef_mask["image"])
        imsave.save(
            os.path.join(
                self.output_directory,
                "maps",
                os.path.splitext(os.path.split(self.imctrlname)[1])[0] + "_predef.tif"
            )
        )
        if (self.flatfield is not None) and (self.flatfield != ""):
            imsave = Image.fromarray(flatfield_image)
            imsave.save(
                os.path.join(
                    self.output_directory,
                    "maps",
                    os.path.splitext(os.path.split(self.imctrlname)[1])[0] + "_flatfield.tif"
                )
            )
        self.cache["Image Controls"] = image_dict["Image Controls"]
        # TODO: Look at image size?
        # img.setControl('pixelSize',[150.0,150.0])
        _, tifdata, _, _ = GetTifData(self.filename)
        image_dict["Image Controls"]["pixelSize"] = tifdata["pixelSize"]
        self.cache["Image Controls"]["pixelSize"] = tifdata["pixelSize"]
        # cache['Masks'] = img.getMasks()
        # self.cache['Masks'] = image_dict['Masks']
        # cache['intMaskMap'] = img.IntMaskMap() # calc mask & TA arrays to save for integrations
        # for k,v in img_copy['Image Controls'].items():
        #    print(k)
        # for k in img.data['Image Controls'].keys():
        #    if k not in img_copy['Image Controls'].keys():
        #        print(k, img.data['Image Controls'][k])
        # Missing 5: size, samplechangerpos, det2theta, ImageTag, formatName
        # [2880,2880] None 0.0 None GSAS-II known TIF image
        # only size seems to be holding anything meaningful at this time, though det2theta and samplechangerpos could hold something later
        if self.stopEarly: return
        
        # self.cache["intMaskMap"] = MakeUseMask(
        #     image_dict["Image Controls"],image_dict["Masks"], blkSize=self.blkSize
        # )
        # cache['intTAmap'] = img.IntThetaAzMap()
        self.cache["intTAmap"] = MakeUseTA(image_dict["Image Controls"],self.blkSize)
        if self.stopEarly: return
        # cache['FrameMask'] = img.MaskFrameMask() # calc Frame mask & T array to save for Pixel masking
        self.cache["FrameMask"] = MaskFrameMask(image_dict)
        if self.stopEarly: return
        # cache['maskTmap'] = img.MaskThetaMap()
        self.cache["maskTmap"] = Make2ThetaAzimuthMap(
            image_dict["Image Controls"],
            (0, image_dict["Image Controls"]["size"][0]),
            (0, image_dict["Image Controls"]["size"][1])
        )[0]
        if self.stopEarly: return
        getmaps(self.cache, self.imctrlname, os.path.join(self.output_directory, "maps"))
        self.cache["AzimMask"] = np.logical_or(self.cache["pixelAzmap"] < self.cache["Image Controls"]["LRazimuth"][0], self.cache["pixelAzmap"] > self.cache["Image Controls"]["LRazimuth"][1])
        if self.stopEarly: return
        # 2th fairly linear along center; calc 2th - pixelsize conversion
        center = self.cache["Image Controls"]["center"]
        center[0] = center[0] * 1000.0 / self.cache["Image Controls"]["pixelSize"][0]
        center[1] = center[1] * 1000.0 / self.cache["Image Controls"]["pixelSize"][1]
        self.cache["center"] = center
        image_dict["center"] = center
        # self.cache['d2th'] = (self.cache['pixelTAmap'][int(center[1]),0] - self.cache['pixelTAmap'][int(center[1]),99])/100
        self.cache["esdMul"] = self.esdMul
        image_dict["Masks"]["SpotMask"]["esdMul"] = self.esdMul
        numChansAzim = 360
        self.cache["azimband"] = get_azimbands(self.cache["pixelAzmap"], numChansAzim)
        if self.stopEarly: return

        # numChans
        LUtth = np.array(self.cache["Image Controls"]["IOtth"])
        wave = self.cache["Image Controls"]["wavelength"]
        dsp0 = wave / (2.0 * sind(LUtth[0] / 2.0))
        dsp1 = wave / (2.0 * sind(LUtth[1] / 2.0))
        x0 = GetDetectorXY2(dsp0, 0.0, self.cache["Image Controls"])[0]
        x1 = GetDetectorXY2(dsp1, 0.0, self.cache["Image Controls"])[0]
        if not np.any(x0) or not np.any(x1):
            raise Exception
        numChans = int(1000 * (x1 - x0) / self.cache["Image Controls"]["pixelSize"][0]) // 2
        self.cache["numChans"] = numChans
        self.cache["Qbins"], self.cache["QbinEdges"] = get_Qbands(self.cache["pixelQmap"], LUtth, wave, numChans)

        # pytorch integration
        (
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        ) = prepare_qmaps(
            self.cache["pixelTAmap"],
            self.cache["polscalemap"],
            self.cache["pixelsampledistmap"],
            self.cache["Image Controls"]["IOtth"][0],
            self.cache["Image Controls"]["IOtth"][1],
            self.cache["Image Controls"]["outChannels"],
        )

        if self.stopEarly: return

        # gradient info
        self.cache["gradient"] = gradient_cache(
            predef_mask["image"].shape, center, np.ones((3, 3), dtype=np.uint)
        )

        self.cache["image_dict"] = image_dict

        cache_time = time.time() - cache_time
        print(cache_time)

        self.finished.emit()


class SingleIterator(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)

    def __init__(
        self,
        cache,
        filename,
        imctrlname,
        imgmaskname,
        input_directory,
        output_directory,
        name,
        number,
        ext,
        closing_method="binary_closing",
        calc_outlier = True,
        outChannels = 0,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        calc_spottiness = False,
        logging=False,
        timing=None,
        timing_names = None,
    ):
        super().__init__()
        self.cache = cache.copy()
        self.filename = filename
        self.imctrlname = imctrlname
        self.imgmaskname = imgmaskname
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.name = name
        self.number = number
        self.ext = ext
        self.closing_method = closing_method
        self.calc_outlier = calc_outlier
        self.outChannels = outChannels
        self.calc_splitting = calc_splitting
        self.azim_Q_shape_min = azim_Q_shape_min
        self.calc_spottiness = calc_spottiness
        self.logging = logging
        self.timing = timing
        self.timing_names = timing_names

    def run(self):
        run_iteration(
            self.filename,
            self.input_directory,
            self.output_directory,
            self.name,
            self.number,
            self.cache,
            self.ext,
            return_steps = False,
            calc_outlier = self.calc_outlier,
            outChannels = self.outChannels,
            calc_splitting = self.calc_splitting,
            azim_Q_shape_min = self.azim_Q_shape_min,
            calc_spottiness = self.calc_spottiness,
            timing = self.timing,
            timing_names = self.timing_names,
        )
        self.finished.emit()


class AdvancedSettings(QtWidgets.QWidget):

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

        self.calc_outlier_checkbox = QtWidgets.QCheckBox("Perform outlier masking")
        self.calc_outlier_default = True
        self.calc_outlier_checkbox.setChecked(self.calc_outlier_default)
        self.calc_outlier_checkbox.checkStateChanged.connect(self.toggle_outlier_settings)
        self.calc_splitting_checkbox = QtWidgets.QCheckBox("Perform spot/texture outlier mask splitting")
        self.calc_splitting_default = True
        self.calc_splitting_checkbox.setChecked(self.calc_splitting_default)
        self.calc_spottiness_checkbox = QtWidgets.QCheckBox("Calculate Spottiness of Rings")
        self.calc_spottiness_default = True
        self.calc_spottiness_checkbox.setChecked(self.calc_spottiness_default)

        self.defaults_button = QtWidgets.QPushButton("Restore Defaults")
        self.defaults_button.released.connect(self.restore_defaults)

        self.outlier_settings = QtWidgets.QWidget()
        self.outlier_layout = QtWidgets.QGridLayout()
        self.outlier_settings.setLayout(self.outlier_layout)
        self.settings_layout = QtWidgets.QGridLayout()
        # self.settings_layout.addWidget(self.settings_label, 0, 0, 1, 2)
        self.settings_layout.addWidget(self.calc_outlier_checkbox, 0, 0, 1, 2)
        self.outlier_layout.addWidget(self.override_label, 0, 0, 1, 2)
        self.outlier_layout.addWidget(self.madmult_override, 1, 0)
        self.outlier_layout.addWidget(self.madmult, 1, 1)
        self.outlier_layout.addWidget(self.nbins_om_override, 2, 0)
        self.outlier_layout.addWidget(self.nbins_om, 2, 1)
        self.outlier_layout.addWidget(self.calc_splitting_checkbox, 3, 0, 1, 2)
        self.outlier_layout.addWidget(self.azim_q_override, 4, 0)
        self.outlier_layout.addWidget(self.azim_q, 4, 1)
        self.outlier_layout.addWidget(self.calc_spottiness_checkbox, 5, 0, 1, 2)
        self.settings_layout.addWidget(self.outlier_settings, 1, 0, 6, 2)
        self.settings_layout.addWidget(self.defaults_button, 7, 0)

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
        self.calc_spottiness_checkbox.setChecked(self.calc_spottiness_default)


class main_window(QtWidgets.QWidget):
    # text box/file browser directory location
    # ditto config file
    # ditto predef mask
    # start button
    # clear queue button
    # optional "choose existing files to run over" section
    # default: none, shortcut button for all, else choose which files
    def __init__(self, input_directory=None, output_directory=None, imctrl=None, flatfield=None, imgmask=None, bad_pixels=None):
        super().__init__()
        # self.directory_text = QtWidgets.QPushButton("Directory:")
        # self.directory_loc = QtWidgets.QLabel()
        self.input_directory_widget = file_select(
            "Input Directory:",
            default_text=input_directory,
            isdir=True,
        )
        self.output_directory_widget = file_select(
            "Output Directory:",
            default_text=output_directory,
            isdir=True,
        )
        # self.config_text = QtWidgets.QPushButton("Config file:")
        # self.config_loc = QtWidgets.QLabel()
        self.config_widget = imctrl_file_select(
            "Config file:",
            default_text=imctrl,
            startdir=self.input_directory_widget.file_name.text(),
            ext="Imctrl and PONI files (*.imctrl *.poni)",
        )
        self.config_widget.imctrl_set.connect(self.update_imctrl_data)
        # self.predef_mask_text = QtWidgets.QPushButton("Predefined Mask:")
        # self.predef_mask_loc = QtWidgets.QLabel()
        self.flatfield_widget = file_select(
            "Flat-field file:",
            default_text=flatfield,
            startdir=self.input_directory_widget.file_name.text()
        )
        self.predef_mask_widget = file_select(
            "Experimental Mask:",
            default_text=imgmask,
            startdir=self.input_directory_widget.file_name.text(),
        )
        self.bad_pixel_mask_widget = file_select(
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
        self.outChannels.setMaximum(10000)
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
        self.process_existing_images_checkbox = QtWidgets.QCheckBox(
            "Process existing images"
        )
        self.regex_label = QtWidgets.QLabel("Regex for existing images:")
        self.existing_images_regex = QtWidgets.QTextEdit()

        self.settings = {}
        self.settings_widget = AdvancedSettings(settings=self.settings)
        self.settings_shown = False

        # self.time_checkpoints = ["Start","Image loaded","Cache","Zero mask","Polar-correct","Outlier mask","Closing mask","Split first mask","Split second mask","All integrations","Save integrals","Delete project"]
        # self.time_checkpoints = ["Start", "Cache", "Zero mask", "Outlier mask", "Closing mask", "Splitting mask", "Integrations", "Save integrals", "CSim", "NMI", "SSim"]
        self.all_times = []
        # set up gsas-ii project
        # G2sc.blkSize = 2**8  # computer-dependent tuning parameter
        self.blkSize = 2**8
        # G2sc.SetPrintLevel('warn')   # reduces output

        self.cache = {}  # place to save intermediate computations

        self.queue = deque()
        self.event_handler = image_monitor(self.queue)
        self.stop_event = threading.Event()
        # self.watchdog_thread = threading.Thread(target=watchdog_observer,args=(self.directory,self.event_handler),daemon=True)

        self.timer = QtCore.QTimer()
        self.keep_running = False
        self.timer.timeout.connect(self.on_timeout)

        # self.iteration_thread = QtCore.QThread()
        # self.cache_thread = QtCore.QThread()

        self.queue_length_info = QtWidgets.QLabel(
            "Queue is {0} items long".format(len(self.queue))
        )

        self.list_of_times = []
        self.list_of_time_names = []

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
        self.window_layout.addWidget(self.input_directory_widget, 0, 0, 1, 3)
        self.window_layout.addWidget(self.output_directory_widget, 1, 0, 1, 3)
        self.window_layout.addWidget(self.config_widget, 2, 0, 1, 3)
        self.window_layout.addWidget(self.poni_config_options, 3, 1, 3, 2)
        self.window_layout.addWidget(self.flatfield_widget, 6, 0, 1, 3)
        self.window_layout.addWidget(self.predef_mask_widget, 7, 0, 1, 3)
        self.window_layout.addWidget(self.bad_pixel_mask_widget, 8, 0, 1, 3)
        self.window_layout.addWidget(self.advanced_settings_button, 10, 0)
        self.window_layout.addWidget(self.start_button, 9, 0)
        self.window_layout.addWidget(self.clear_queue_button, 9, 1)
        self.window_layout.addWidget(self.stop_button, 9, 2)
        self.window_layout.addWidget(self.settings_widget, 11, 0, 1, 3)
        self.window_layout.addWidget(self.process_existing_images_checkbox, 12, 0)
        self.window_layout.addWidget(self.queue_length_info, 13, 0)
        # self.window_layout.addWidget(self.regex_label,7,0)
        # self.window_layout.addWidget(self.existing_images_regex,8,0)
        self.settings_widget.hide()

        self.setLayout(self.window_layout)
        self.show()

    def update_imctrl_data(self):
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

    def cache_thread_finished(self):
        self.has_made_cache = True
        self.cache_thread = None
        if self.keep_running:
            self.timer.start()

    def iteration_thread_finished(self):
        if self.keep_running:
            self.timer.start()

    def on_timeout(self):
        if self.keep_running:
            # block=True in Queue.get() tells it to wait until there is something in the queue to grab it
            # can also set a timeout value (in seconds) to wait before it throws an Empty exception
            # Windows systems apparently have a problem with block=True, timeout=None
            # filename,name,number = queue.get(block=True,timeout=30)
            if self.queue:
                self.queue_length_info.setText(
                    "Queue is {0} items long".format(len(self.queue))
                )
                if self.has_made_cache:
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
                                self.cache,
                                filename,
                                self.imgctrl,
                                self.imgmask,
                                self.input_directory,
                                self.output_directory,
                                name,
                                number,
                                ext,
                                calc_outlier = self.settings_widget.calc_outlier_checkbox.isChecked(),
                                calc_splitting = self.settings_widget.calc_splitting_checkbox.isChecked(),
                                calc_spottiness = self.settings_widget.calc_spottiness_checkbox.isChecked(),
                                timing = self.list_of_times,
                                timing_names = self.list_of_time_names,
                            )
                        else:
                            self.iteration_worker = SingleIterator(
                                self.cache,
                                filename,
                                self.imgctrl,
                                self.imgmask,
                                self.input_directory,
                                self.output_directory,
                                name,
                                number,
                                ext,
                                azim_Q_shape_min = self.settings_widget.azim_q.value(),
                                calc_outlier = self.settings_widget.calc_outlier_checkbox.isChecked(),
                                calc_splitting = self.settings_widget.calc_splitting_checkbox.isChecked(),
                                calc_spottiness = self.settings_widget.calc_spottiness_checkbox.isChecked(),
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
                        print(filename)
                        esdMul = self.settings_widget.madmult_default
                        if self.settings_widget.madmult_override.isChecked():
                            esdMul = self.settings_widget.madmult.value()
                        not_in_poni_settings = {}
                        if self.iotth_max.value() != 0.0:
                            not_in_poni_settings["IOtth"] = [
                                self.iotth_min.value(),
                                self.iotth_max.value()
                            ]
                        if (self.azim_min.value() != 0.0) or (os.path.splitext(self.imgctrl)[1] == ".poni"):
                            not_in_poni_settings["LRazimuth"] = [
                                self.azim_min.value(),
                                self.azim_max.value()
                            ]
                            print(f"azim range: {not_in_poni_settings['LRazimuth']}")
                        if self.outChannels.value() != 0.0:
                            not_in_poni_settings["outChannels"] = self.outChannels.value()
                        if self.PolaVal.value() != 0.0:
                            not_in_poni_settings["PolaVal"] = [self.PolaVal.value(), False]
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
                            not_in_poni_settings = not_in_poni_settings,
                        )
                        self.cache_worker.moveToThread(self.cache_thread)
                        self.cache_thread.started.connect(self.cache_worker.run)
                        self.cache_worker.finished.connect(self.cache_thread.quit)
                        self.cache_worker.finished.connect(
                            self.cache_worker.deleteLater
                        )
                        # self.cache_thread.finished.connect(self.cache_thread.deleteLater)
                        self.cache_thread.finished.connect(self.cache_thread_finished)
                        self.cache_thread.start()
            else:
                self.queue_length_info.setText("Queue is 0 items long")
            # else:
            #    #If it's been over an hour since the last update, stop
            #    if time.time() - wait_start > 60:
            #        print("Average time: {0}".format(np.average(self.all_times)))
            #        self.keep_running = False
        else:
            self.timer.stop()

    def start_processing(self):
        self.input_directory = self.input_directory_widget.file_name.text()
        self.output_directory = self.output_directory_widget.file_name.text()
        self.imgctrl = self.config_widget.file_name.text()
        self.imgmask = self.predef_mask_widget.file_name.text()
        self.flatfield = self.flatfield_widget.file_name.text()
        self.bad_pixels = self.bad_pixel_mask_widget.file_name.text()
        self.cache = {}
        self.has_made_cache = False
        # print("Directory: {0}, Ctrl file: {1}, Predef mask: {2}".format(dir_name,ctrl_name,predef_mask))
        # self.process = main_process(dir_name,ctrl_name,predef_mask)
        # create subdirectories if needed
        newdirs = ["maps", "masks", "integrals", "stats", "grads"]
        if not ((self.flatfield is None) or (self.flatfield == "")):
            newdirs.append("flatfield")
        for newdir in newdirs:
            path = os.path.join(self.output_directory, newdir)  # store maps with the images
            if not os.path.exists(path):
                os.mkdir(path)

        # Grab existing file names and add them to the queue if option checked
        if self.process_existing_images_checkbox.isChecked():
            # existing_files = glob.glob(self.directory+"/*.metadata")
            existing_files = glob.glob(self.input_directory + "/*.tif")
            # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif.metadata$"
            # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif$"
            reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
            # number -> actual int
            # number_int = results[0].group("number").remove("-").remove("_")
            # number_int = int(number_int)
            for filename in existing_files:
                # Add file path to queue, stripping ".metadata"
                results = re.match(reg_image, filename)
                # Regex observer uses re.findall(), so it needs results[0].
                # self.queue.append([filename[:-9],results.group("name"),results.group("number")])
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

        self.observer = Observer()
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
            "Queue is {0} items long".format(len(self.queue))
        )

    def pause(self):
        print("Pausing. If processing an image, that process will complete first.")
        self.keep_running = False
        # watchdog thread will still keep populating the queue

    def resume(self):
        self.keep_running = True
        self.timer.start(100)

    def update_dir(self, input_directory):
        self.pause()
        self.clear_queue()
        self.input_directory = input_directory
        # self.watchdog_thread = threading.Thread(target=watchdog_observer,args=(self.directory,self.event_handler),daemon=True)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.input_directory, recursive=False)
        self.observer.start()
        # self.resume()

    def advanced_settings_button_pressed(self):
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
        print("Stopping and clearing queue")
        # print(f"Length of timing list: {len(self.list_of_times)}")
        # print(f"Mean time: {np.mean(self.list_of_times):.4f} +/- {np.std(self.list_of_times):.4f}")
        means = np.mean(self.list_of_times, axis=0)
        std = np.std(self.list_of_times, axis=0)
        for i in range(len(self.list_of_time_names)):
            print(f"{self.list_of_time_names[i]}: {means[i]:.4f} +/- {std[i]:.4f}")
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

    def really_stopped(self):
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

    def closeEvent(self, evt):
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


# Following https://gsas-ii.readthedocs.io/en/latest/GSASIIscriptable.html 16.7.9. Optimized Image Integration
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory")
    parser.add_argument("-o", "--output_directory")
    parser.add_argument("-c", "--imctrl")
    parser.add_argument("-f", "--flatfield", default=None)
    parser.add_argument("-m", "--imgmask", default=None)
    parser.add_argument("-b", "--bad_pixels", default=None)
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
    window = main_window(input_directory=input_directory, output_directory=output_directory, imctrl=imgctrl, flatfield=flatfield, imgmask=imgmask, bad_pixels=bad_pixels)
    sys.exit(app.exec())
