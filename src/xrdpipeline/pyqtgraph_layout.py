import glob
import sys
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import PySide6
import pyqtgraph as pg
import tifffile as tf
from pyqtgraph.Qt import QtCore, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major")


def tth_to_q(tth, wavelength):
    return 4 * np.pi * np.sin(tth / 2 * np.pi / 180) / wavelength


def q_to_tth(q, wavelength):
    return np.arcsin(q * wavelength / (4 * np.pi)) * (360 / np.pi)


def tth_to_d(tth, wavelength):
    return wavelength / (2 * np.sin(tth / 2 * np.pi / 180))


# global tiflist, keylist, curr_pos, curr_key


@dataclass
class ColorSettings:
    name: str
    color: str
    default_color: str = field(init=False)

    def __post_init__(self):
        self.default_color = self.color


@dataclass
class Settings:
    directory: str
    imagecontrol: str
    image_size: tuple
    wavelength: float
    outChannels: int
    keylist: list
    tiflist: dict
    curr_key: int = 0
    curr_pos: int = 0
    # colors: ColorsList = ColorsList()
    colors: dict = field(
        default_factory=lambda: {
            "predef_mask": ColorSettings("Predefined Mask", "hotpink"),
            "nonpositive_mask": ColorSettings("Nonpositive Mask", "purple"),
            "base_line": ColorSettings("Base Integral Line", "black"),
            "outlier_mask": ColorSettings("Outlier Mask", "lime"),
            "outlier_line": ColorSettings("Outlier Masked Integral Line", "lime"),
            "spot_mask": ColorSettings("Spot Mask", "cyan"),
            "spot_line": ColorSettings("Spot Masked Integral Line", "cyan"),
            "arcs_mask": ColorSettings("Texture Mask", "red"),
            "arcs_line": ColorSettings("Texture Masked Integral Line", "red"),
            "minus_spot_line": ColorSettings(
                "Base \u2013 Spot Masked Integral Line", "black"
            ),
            "minus_arcs_line": ColorSettings(
                "Base \u2013 Texture Masked Integral Line", "black"
            ),
            "tth_circle_mask": ColorSettings("2Theta Circle", "white"),
        }
    )


class ColorWidget(QtWidgets.QWidget):
    def __init__(self, coloritem: ColorSettings):
        super().__init__()
        self.coloritem = coloritem
        self.name_label = QtWidgets.QLabel(self.coloritem.name + ":")
        self.color_text = QtWidgets.QLineEdit(self.coloritem.color)
        # self.current_color_preview =
        self.widgetlayout = QtWidgets.QHBoxLayout()
        self.widgetlayout.addWidget(self.name_label)
        self.widgetlayout.addWidget(self.color_text)
        self.setLayout(self.widgetlayout)

    def apply(self):
        self.coloritem.color = self.color_text.text()


class SettingsWindow(QtWidgets.QWidget):
    # apply_settings = pg.QtCore.pyqtSignal()
    apply_settings = pg.QtCore.Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

        self.tabs = QtWidgets.QTabWidget()
        self.base_settings = QtWidgets.QWidget()
        self.color_settings = QtWidgets.QWidget()
        self.tabs.addTab(self.base_settings, "Image Controls")
        self.tabs.addTab(self.color_settings, "Colors")

        self.directory_label = QtWidgets.QLabel("Directory:")
        self.directory_text = QtWidgets.QLineEdit(self.settings.directory)
        self.directory_browse_button = QtWidgets.QPushButton("Browse...")

        self.imctrl_file_label = QtWidgets.QLabel("Image Control File:")
        self.imctrl_file_text = QtWidgets.QLineEdit(self.settings.imagecontrol)
        self.imctrl_file_browse_button = QtWidgets.QPushButton("Browse...")

        self.wavelength_label = QtWidgets.QLabel("Wavelength:")
        self.wavelength_text = QtWidgets.QLineEdit(str(self.settings.wavelength))

        self.outChannels_label = QtWidgets.QLabel("Number of Integration Bins:")
        self.outChannels_text = QtWidgets.QLineEdit(str(self.settings.outChannels))

        self.apply_button = QtWidgets.QPushButton("Apply")
        self.okay_button = QtWidgets.QPushButton("Apply and Close")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

        # colors
        self.colors_layout = QtWidgets.QVBoxLayout()
        self.color_widgets = {}
        for name, coloritem in self.settings.colors.items():
            self.color_widgets[name] = ColorWidget(coloritem)
            self.colors_layout.addWidget(self.color_widgets[name])
        self.default_colors_button = QtWidgets.QPushButton("Return to Defaults")
        self.colors_layout.addWidget(self.default_colors_button)
        self.default_colors_button.released.connect(self.default_colors_pressed)

        self.settings_layout = QtWidgets.QGridLayout()
        self.settings_layout.addWidget(self.directory_label, 0, 0)
        self.settings_layout.addWidget(self.directory_text, 0, 1)
        self.settings_layout.addWidget(self.directory_browse_button, 0, 2)
        self.settings_layout.addWidget(self.imctrl_file_label, 1, 0)
        self.settings_layout.addWidget(self.imctrl_file_text, 1, 1)
        self.settings_layout.addWidget(self.imctrl_file_browse_button, 1, 2)
        self.settings_layout.addWidget(self.wavelength_label, 2, 0)
        self.settings_layout.addWidget(self.wavelength_text, 2, 1, 1, 2)
        self.settings_layout.addWidget(self.outChannels_label, 3, 0)
        self.settings_layout.addWidget(self.outChannels_text, 3, 1, 1, 2)

        self.settings_layout.setColumnStretch(0, 2)
        self.settings_layout.setColumnStretch(1, 5)
        self.settings_layout.setColumnStretch(2, 0)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.apply_button)
        self.button_layout.addWidget(self.okay_button)
        self.button_layout.addWidget(self.cancel_button)

        self.base_settings.setLayout(self.settings_layout)
        self.color_settings.setLayout(self.colors_layout)

        self.main_layout = QtWidgets.QVBoxLayout()
        # self.main_layout.addLayout(self.settings_layout)
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)

        self.directory_browse_button.released.connect(self.browse_dir)
        self.imctrl_file_browse_button.released.connect(self.browse_imctrl)
        self.apply_button.released.connect(self.apply_button_pressed)
        self.okay_button.released.connect(self.okay_button_pressed)
        self.cancel_button.released.connect(self.cancel_button_pressed)

    def update_shown_info(self):
        self.directory_text.setText(self.settings.directory)
        self.imctrl_file_text.setText(self.settings.imagecontrol)
        self.wavelength_text.setText(str(self.settings.wavelength))
        self.outChannels_text.setText(str(self.settings.outChannels))

    def browse_dir(self):
        directory_name = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Image Directory"
        )
        self.directory_text.setText(directory_name)

    def browse_imctrl(self):
        imctrl_file_name = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose Configuration File",
            self.directory_text.text(),
            "Imctrl and PONI files (*.imctrl *.poni)",
        )
        self.imctrl_file_text.setText(imctrl_file_name[0])
        # read in wavelength and integral bins from file
        if ".imctrl" in imctrl_file_name[0]:
            with open(imctrl_file_name[0], "r") as infile:
                filetext = infile.read()
            matches = re.findall("wavelength:([\d\.]+)", filetext)
            self.wavelength_text.setText(matches[0])
            matches = re.findall("outChannels:([\d.]+)", filetext)
            self.outChannels_text.setText(matches[0])
            # print(matches[0])
        elif ".poni" in imctrl_file_name[0]:
            with open(imctrl_file_name[0], "r") as infile:
                filetext = infile.read()
            matches = re.findall("Wavelength: ([\d.e+-]+)", filetext)
            self.wavelength_text.setText(matches[0]) * (10**10)
            self.outChannels_text.setText(1000)  # pick a default value

    def default_colors_pressed(self):
        for name, coloritem in self.settings.colors.items():
            self.color_widgets[name].color_text.setText(coloritem.default_color)

    def apply_changes(self):
        self.settings.directory = self.directory_text.text()
        self.settings.imagecontrol = self.imctrl_file_text.text()
        self.settings.wavelength = float(self.wavelength_text.text())
        self.settings.outChannels = int(self.outChannels_text.text())
        for name, coloritem in self.settings.colors.items():
            coloritem.color = self.color_widgets[name].color_text.text()
        self.apply_settings.emit()

    def apply_button_pressed(self):
        self.apply_changes()

    def okay_button_pressed(self):
        self.apply_changes()
        # self.close(is_from_button=True)
        self.close()

    def cancel_button_pressed(self):
        # self.close(is_from_button=True)
        self.close()

    # def closeEvent(self,evt,is_from_button=False):
    #     print(evt)
    #     print(evt.sender())
    #     if is_from_button:
    #         evt.accept()
    #     else:
    #         answer = QtWidgets.QMessageBox.question(self,"Exit","Would you like to apply any changes?",QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No | QtWidgets.QMessageBox.StandardButton.Cancel, QtWidgets.QMessageBox.StandardButton.Cancel)
    #         if answer == QtWidgets.QMessageBox.StandardButton.Yes:
    #             self.apply_settings.emit()
    #             evt.accept()
    #         elif answer == QtWidgets.QMessageBox.StandardButton.No:
    #             evt.accept()
    #         else:
    #             evt.ignore()


class KeyPressWindow(QtWidgets.QWidget):
    def __init__(self, directory=".", imagecontrol=""):
        super().__init__()
        # global tiflist, keylist, curr_key, curr_pos
        # self.curr_key = 0
        # self.curr_pos = 0
        # self.keylist = []
        # self.tiflist = {}
        # self.settings = {
        #     "directory": directory,
        #     "imagecontrol": imagecontrol,
        #     "image_size": [2880,2880],
        #     "wavelength": 0,
        #     "keylist": self.keylist,
        #     "tiflist": self.tiflist,
        #     # "curr_key": self.curr_key, # need these to be pointers; lists and dicts do that already, but ints do not
        #     # "curr_pos": self.curr_pos
        # }
        self.settings = Settings(
            directory, imagecontrol, (2880, 2880), 0, 0, [], {}, 0, 0
        )
        self.settings_widget = SettingsWindow(self.settings)
        self.settings_widget.apply_settings.connect(self.update_dir)
        # self.directory = directory
        # self.update_tiflist()
        self.menubar = pg.QtWidgets.QMenuBar()
        self.menubar.setMaximumHeight(40)
        self.filemenu = self.menubar.addMenu("&File")
        self.filemenu.addAction(
            "Load Directory and Image Control File", self.choose_dir
        )
        self.settingsmenu = self.menubar.addMenu("&Settings")
        self.settingsmenu.addAction("Open Settings...", self.show_settings)
        self.imageview = MainImageView(self.settings, parent=self)
        self.integral_widget = IntegralView(self, self.settings)
        self.tabbed_area = TabbedArea(self, self.settings)

        # Show name of image
        # self.name_label = QtWidgets.QLabel("<span style='font-size: 12pt'>{0}</span>".format(tiflist[keylist[curr_key]][curr_pos]))
        self.name_label = QtWidgets.QLabel()
        self.name_label.setMinimumSize(500, 50)

        # Placing the mouseMoved() event here so it can more easily be passed to the integral widget
        self.cursor_label = QtWidgets.QLabel(
            "<span style='font-size: 12pt'>x=%0.0f,   y=%0.0f</span>,   <span style='color: red; font-size: 12pt'>2Theta=%0.1f,    Azim=%0.1f,</span>    <span style='font-size: 12pt'>Q=%0.0f,    d=%0.0f</span>"
            % (0, 0, 0, 0, 0, 0)
        )
        self.cursor_label.setMinimumSize(500, 50)
        self.integral_cursor_label = pg.TextItem(text="tth= \nq= \nd= ", anchor=(0, 1))
        self.integral_widget.integral_view.addItem(
            self.integral_cursor_label, ignoreBounds=True
        )
        self.imageview.view.scene().sigMouseMoved.connect(self.mouseMovedImage)
        self.integral_widget.integral_view.scene().sigMouseMoved.connect(
            self.mouseMovedIntegral
        )
        # self.integral_widget.integral_view.scene().sigMouseHover.connect(self.mouseHoverIntegral)
        self.tabbed_area.contour_widget.view.scene().sigMouseMoved.connect(
            self.mouseMovedContour
        )
        # self.tabbed_area.stats_widget.stats_view.scene().sigMouseMoved.connect(self.mouseMovedStats)
        self.tabbed_area.contour_widget.view.scene().sigMouseClicked.connect(
            self.mouseClickedContourChangeImage
        )
        # checkboxes for the vertical line and circle
        self.vLineCheckbox = QtWidgets.QCheckBox("2Th Line")
        # self.vLineCheckbox.setChecked(True)
        self.vLineCheckbox.stateChanged.connect(self.vLineCheckbox_changed)
        self.circleCheckbox = QtWidgets.QCheckBox("2Th Circle")
        # self.circleCheckbox.setChecked(True)
        self.circleCheckbox.stateChanged.connect(self.circleCheckbox_changed)
        self.linked_axes_checkbox = QtWidgets.QCheckBox("Link X axes")
        self.linked_axes_checkbox.stateChanged.connect(
            self.linked_axes_checkbox_changed
        )

        self.live_view_image_checkbox = QtWidgets.QCheckBox("Live update")
        self.live_view_image_checkbox.stateChanged.connect(
            self.live_view_image_checkbox_changed
        )
        self.x_axis_choice = QtWidgets.QComboBox()
        self.x_axis_types = ["2 Theta", "Q"]
        self.x_axis_choice.addItems(self.x_axis_types)
        self.x_axis_choice.setCurrentIndex(0)
        self.x_axis_choice.currentIndexChanged.connect(self.x_axis_changed)
        # self.tooltip = QtWidgets.QToolTip()
        # self.tooltip.isVisible()

        # win.show()
        self.layout = QtWidgets.QGridLayout()
        # addWidget(widget,row,col,row_length,col_length)
        self.layout.addWidget(self.menubar, 0, 0, 1, 11)
        self.layout.addWidget(self.imageview, 1, 0, 7, 5)
        self.layout.addWidget(self.name_label, 8, 0, 1, 4)
        self.layout.addWidget(self.live_view_image_checkbox, 8, 4)
        self.layout.addWidget(self.cursor_label, 9, 0, 1, 4)
        self.layout.addWidget(self.imageview.predef_mask_box, 10, 0)
        self.layout.addWidget(self.imageview.mask_box, 10, 1)
        self.layout.addWidget(self.imageview.spot_mask_box, 10, 2)
        self.layout.addWidget(self.imageview.arcs_mask_box, 10, 3)
        self.layout.addWidget(self.circleCheckbox, 12, 0)
        self.layout.addWidget(self.imageview.mask_opacity_label, 12, 1)
        self.layout.addWidget(self.imageview.mask_opacity_box, 12, 2)
        # blank space to help ameliorate formatting... should set sizes for other widgets
        self.layout.addWidget(QtWidgets.QWidget(), 13, 0)

        self.layout.addWidget(self.integral_widget, 1, 5, 2, 6)
        self.layout.addWidget(self.integral_widget.base_integral_checkbox, 3, 5)
        self.layout.addWidget(self.integral_widget.masked_integral_checkbox, 3, 6)
        self.layout.addWidget(self.integral_widget.spotmasked_integral_checkbox, 3, 7)
        self.layout.addWidget(
            self.integral_widget.texturemasked_integral_checkbox, 3, 8
        )
        self.layout.addWidget(self.integral_widget.spots_diff_integral_checkbox, 3, 9)
        self.layout.addWidget(self.integral_widget.arcs_diff_integral_checkbox, 3, 10)
        # other
        self.layout.addWidget(self.integral_widget.integral_offset_text, 4, 5)
        self.layout.addWidget(self.integral_widget.integral_offset, 4, 6)
        self.layout.addWidget(self.integral_widget.sqrt_checkbox, 4, 7)
        self.layout.addWidget(self.x_axis_choice, 4, 8)
        self.layout.addWidget(self.vLineCheckbox, 5, 5)
        self.layout.addWidget(self.linked_axes_checkbox, 5, 6)

        self.layout.addWidget(self.tabbed_area, 6, 5, 8, 7)

        self.setLayout(self.layout)
        self.show()

        self.timer = QtCore.QTimer()

    # def start_cycling(self,tick=100):
    #    self.timer.start(tick) #interval in ms
    # def stop_cycling(self):
    #    self.timer.stop()

    def show_settings(self):
        # settings_widget = SettingsWindow(self.settings)
        self.settings_widget.update_shown_info()
        self.settings_widget.show()

    def choose_dir(self):
        directory_name = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Image Directory"
        )
        imctrl_file_name = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose Configuration File",
            directory_name,
            "Imctrl and PONI files (*.imctrl *.poni)",
        )
        self.settings.directory = directory_name
        self.settings.imagecontrol = imctrl_file_name[0]
        self.update_dir()

    def update_dir(self):
        self.settings.curr_key = 0
        self.settings.curr_pos = 0
        if ".imctrl" in self.settings.imagecontrol:
            with open(self.settings.imagecontrol, "r") as infile:
                filetext = infile.read()
            matches = re.findall("wavelength:([\d\.]+)", filetext)
            self.settings.wavelength = float(matches[0])
            matches = re.findall("outChannels:([\d.]+)", filetext)
            self.settings.outChannels = int(matches[0])
            # print(matches[0])
        elif ".poni" in self.settings.imagecontrol:
            with open(self.settings.imagecontrol, "r") as infile:
                filetext = infile.read()
            matches = re.findall("Wavelength: ([\d.e+-]+)", filetext)
            self.settings.wavelength = float(matches[0]) * (10**10)
        self.update_tiflist()
        self.imageview.update_dir()
        self.integral_widget.update_dir()
        self.tabbed_area.update_dir()
        self.name_label.setText(
            "<span style='font-size: 12pt'>{0}</span>".format(
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][
                    self.settings.curr_pos
                ]
            )
        )

    def update_tiflist(self):
        # global tiflist, keylist, curr_key, curr_pos
        fulltiflist = sorted(
            glob.glob(self.settings.directory + "/*.tif"), key=os.path.getctime
        )
        keylist = []
        tiflist = {}
        for tif in fulltiflist: # Probably losing sort here
            #key = tif.split("\\")[-1].split("/")[-1].split("-")[0] # grab label at start of file name, eg "Sam4". Need to work on this, as some are things like "Dewen-4"
            key = os.path.split(tif)[1]
            key = key[0 : re.search(r"(\d{5})", key).start(0)]
            if key not in keylist:
                keylist.append(key)
                tiflist[key] = []
            #initialimage = tif[0:re.search(r'(\d+)\D+$', tif).end(1)] # string from the start to the end of the last set of numbers.
            initialimage = os.path.splitext(os.path.split(tif)[1])[0]
            if initialimage not in tiflist[key]:
                tiflist[key].append(initialimage)
        self.settings.keylist = keylist
        self.settings.tiflist = tiflist
        # get size of first image, if it exists
        self.settings.image_size = self.get_image_size(
            os.path.join(
                self.settings.directory,
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]+".tif"
            )
        )

    def get_image_size(self, imagename):
        image = tf.imread(imagename)
        return image.shape

    def check_for_updates(self):
        # global tiflist, keylist, curr_key, curr_pos
        # First thing saved: zero mask
        # Last things saved: integrals
        # Check integral list against tiflist
        self.update_tiflist()
        self.tabbed_area.contour_widget.update_integral_list()
        last_completed = self.tabbed_area.contour_widget.integral_filelist[-1]
        # print(last_completed.split("\\")[1].split(self.tabbed_area.contour_widget.integral_extension)[0])
        last_completed = os.path.splitext(os.path.split(last_completed)[1])[0]
        # print(tiflist[keylist[curr_key]])
        completed_pos = self.settings.tiflist[
            self.settings.keylist[self.settings.curr_key]
        ].index(last_completed)
        # print(completed_pos)
        if completed_pos != self.settings.curr_pos:
            self.settings.curr_pos = completed_pos
            self.updateImages()

    def updateImages(self, z_reset=False):
        # global curr_pos
        self.name_label.setText(
            "<span style='font-size: 12pt'>{0}</span>".format(
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][
                    self.settings.curr_pos
                ]
            )
        )
        self.imageview.update_image_data(z_reset=z_reset)
        self.imageview.update_masks_data()
        self.integral_widget.update_integral_data()
        self.tabbed_area.contour_widget.horiz_line.setValue(self.settings.curr_pos)
        self.tabbed_area.stats_widget.update_stats_data()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.forward()
        elif event.key() == QtCore.Qt.Key_Left:
            self.backward()
        elif event.key() == QtCore.Qt.Key_Up:
            self.prevkey()
        elif event.key() == QtCore.Qt.Key_Down:
            self.nextkey()
        elif event.key() == QtCore.Qt.Key_Space:
            self.update_tiflist()
        # elif event.key() == Qt.Key_F:
        #     self.start_cycling()
        # elif event.key() == Qt.Key_G:
        #     self.stop_cycling()
        # else:
        #    print("Only Left, Right, Up, Down and Space keys are functional!")

    def forward(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos += 1
        self.settings.curr_pos = self.settings.curr_pos % len(
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]]
        )
        self.updateImages()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def backward(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos -= 1
        self.settings.curr_pos = self.settings.curr_pos % len(
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]]
        )
        self.updateImages()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def prevkey(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos = 0
        self.settings.curr_key -= 1
        self.settings.curr_key = self.settings.curr_key % len(self.settings.keylist)
        self.updateImages(z_reset=True)
        self.tabbed_area.contour_widget.reset_integral_data()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def nextkey(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos = 0
        self.settings.curr_key += 1
        self.settings.curr_key = self.settings.curr_key % len(self.settings.keylist)
        self.updateImages(z_reset=True)
        self.tabbed_area.contour_widget.reset_integral_data()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def mouseMovedImage(self, evt):
        pos = evt
        if self.imageview.view.sceneBoundingRect().contains(pos):
            mousePoint = self.imageview.view.vb.mapSceneToView(pos)
            x_val = int(mousePoint.x())
            y_val = int(mousePoint.y())
            if (
                x_val > 0
                and x_val < self.imageview.image_data.shape[1]
                and y_val > 0
                and y_val < self.imageview.image_data.shape[0]
            ):
                tth = self.imageview.tth_map[y_val, x_val]
                azim = self.imageview.azim_map[y_val, x_val]
                if self.settings.wavelength != 0:
                    # Q = 4*np.pi*np.sin(tth/2 * np.pi/180) / self.wavelength #inverse angstroms
                    Q = tth_to_q(tth, self.settings.wavelength)
                    # d = self.wavelength / (2 * np.sin(tth/2 * np.pi/180)) #angstroms
                    d = tth_to_d(tth, self.settings.wavelength)
                else:
                    Q = 0
                    d = 0
                # calc intensity of underlying image
                z = self.imageview.image_data[int(mousePoint.y()), int(mousePoint.x())]
                self.cursor_label.setText(
                    "<span style='font-size: 12pt'>x=%0.0f,   y=%0.0f,    z=%0.0f</span>,   <span style='color: red; font-size: 12pt'>2Theta=%0.2f,    Azim=%0.1f,</span>    <span style='font-size: 12pt'>Q=%0.2f,    d=%0.2f</span>"
                    % (mousePoint.x(), mousePoint.y(), z, tth, azim, Q, d)
                )
                if self.vLineCheckbox.isChecked():
                    integral_point = mousePoint
                    integral_point.setY(
                        self.integral_widget.integral_view.getAxis("left").range[0]
                    )
                    if self.x_axis_choice.currentIndex() == 0:
                        self.integral_widget.vLine.setPos(tth)
                        integral_point.setX(tth)
                        self.integral_cursor_label.setPos(integral_point)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = tth_to_q(tth, self.settings.wavelength)
                        self.integral_widget.vLine.setPos(Q)
                        integral_point.setX(Q)
                        self.integral_cursor_label.setPos(integral_point)
                if self.circleCheckbox.isChecked():
                    self.imageview.update_tth_circle(tth)
                if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
                    if self.x_axis_choice.currentIndex() == 0:
                        self.tabbed_area.contour_widget.tth_line.setPos(tth)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = tth_to_q(tth, self.settings.wavelength)
                        self.tabbed_area.contour_widget.tth_line.setPos(Q)
                # if self.tabbed_area.stats_line_checkbox.isChecked():
                #    self.tabbed_area.stats_widget.stats_line.setPos(tth)

    def mouseMovedIntegral(self, evt):
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.stats_line_checkbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
        pos = evt
        if self.integral_widget.integral_view.sceneBoundingRect().contains(pos):
            mousePoint = self.integral_widget.integral_view.vb.mapSceneToView(pos)
            if self.vLineCheckbox.isChecked():
                self.integral_cursor_label.setPos(mousePoint)
                if self.x_axis_choice.currentIndex() == 0:
                    tth = mousePoint.x()
                    self.integral_widget.vLine.setPos(tth)
                    # Q = 4*np.pi*np.sin(tth/2 * np.pi/180) / self.wavelength
                    # d = self.wavelength / (2 * np.sin(tth/2 * np.pi/180))
                    Q = tth_to_q(tth, self.settings.wavelength)
                elif self.x_axis_choice.currentIndex() == 1:
                    Q = mousePoint.x()
                    self.integral_widget.vLine.setPos(Q)
                    tth = q_to_tth(Q, self.settings.wavelength)
                d = tth_to_d(tth, self.settings.wavelength)
                self.integral_cursor_label.setText(
                    "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                )
            if self.circleCheckbox.isChecked():
                if self.x_axis_choice.currentIndex() == 0:
                    self.imageview.update_tth_circle(mousePoint.x())
                elif self.x_axis_choice.currentIndex() == 1:
                    self.imageview.update_tth_circle(
                        q_to_tth(mousePoint.x(), self.settings.wavelength)
                    )
            if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
                # axes for both are swapped at the same time, so it can still use mousePoint.x()
                self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
            # if self.tabbed_area.stats_line_checkbox.isChecked():
            #    self.tabbed_area.stats_widget.stats_line.setPos(mousePoint.x())
            # self.tooltip.showText(mousePoint,"test")
            # print(mousePoint)
            # pos.mapToGlobal() does not exist
            # evt.screenPos() does not exist
            # self.hoverpos = pos
            # QtWidgets.QToolTip.showText(self.hoverpos.toPoint(),"test")
            # self.integral_widget.integral_view.setToolTip("test")

    # def mouseHoverIntegral(self,evt):
    #    print(self.hoverpos.toPoint())
    #    #print(self.hoverpos.toPoint(),self.hoverpos.toPoint().screenPos())
    #    QtWidgets.QToolTip.showText(self.hoverpos.toPoint(),"test")

    def mouseMovedContour(self, evt):
        if (
            self.vLineCheckbox.isChecked()
            or self.circleCheckbox.isChecked()
            or self.tabbed_area.contour_widget.tth_line_checkbox.isChecked()
        ):
            pos = evt
            if self.tabbed_area.contour_widget.view.sceneBoundingRect().contains(pos):
                mousePoint = self.tabbed_area.contour_widget.view.vb.mapSceneToView(pos)
                integral_point = mousePoint
                # integral_point.setY(self.integral_widget.integral_view.getAxis("left").range[1]*.85) #near top
                # integral_point.setY(0)
                # axis = self.integral_widget.integral_view.getAxis("left")
                # integral_point.setY(axis.range[0] + (axis.range[1]-axis.range[0])*.05)
                integral_point.setY(
                    self.integral_widget.integral_view.getAxis("left").range[0]
                )
                self.integral_cursor_label.setPos(integral_point)
                if self.vLineCheckbox.isChecked():
                    self.integral_widget.vLine.setPos(mousePoint.x())
                    if self.x_axis_choice.currentIndex() == 0:
                        tth = mousePoint.x()
                        Q = tth_to_q(tth, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = mousePoint.x()
                        tth = q_to_tth(Q, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    self.integral_cursor_label.setText(
                        "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                    )
                if self.circleCheckbox.isChecked():
                    if self.x_axis_choice.currentIndex() == 0:
                        self.imageview.update_tth_circle(mousePoint.x())
                    elif self.x_axis_choice.currentIndex() == 1:
                        self.imageview.update_tth_circle(
                            q_to_tth(mousePoint.x(), self.settings.wavelength)
                        )
                if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
                    self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())

    # def mouseMovedStats(self,evt):
    #    if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.stats_line_checkbox.isChecked():
    #        pos = evt
    #        if self.tabbed_area.stats_widget.stats_view.sceneBoundingRect().contains(pos):
    #            mousePoint = self.tabbed_area.stats_widget.stats_view.vb.mapSceneToView(pos)
    #            if self.vLineCheckbox.isChecked():
    #                self.integral_widget.vLine.setPos(mousePoint.x())
    #            if self.circleCheckbox.isChecked():
    #                self.imageview.update_tth_circle(mousePoint.x())
    #            if self.tabbed_area.stats_line_checkbox.isChecked():
    #                self.tabbed_area.stats_widget.stats_line.setPos(mousePoint.x())

    def mouseClickedContourChangeImage(self, evt):
        # global tiflist, keylist, curr_key, curr_pos
        # evt.button() results = {1: left click, 2: right click, 4: middle click}
        # print(evt.button())
        # If left click
        # Can also check for double-click (evt.double() == True)
        # no longer outputting same evt.button() results
        # if evt.button() == 1:
        #     print("Recognized left click")
        #     pos = int(self.tabbed_area.contour_widget.view.vb.mapSceneToView(evt.scenePos()).y())
        #     if (pos >= 0) and (pos >= self.tabbed_area.contour_widget.view.getAxis("left").range[0]) and (pos < len(self.settings.tiflist[self.settings.keylist[self.settings.curr_key]])):
        #         if self.timer.isActive():
        #             self.pause()
        #         self.curr_pos = pos
        #         print(pos)
        #         self.tabbed_area.contour_widget.horiz_line.setValue(pos)
        #         self.updateImages()
        if evt.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            pos = int(
                self.tabbed_area.contour_widget.view.vb.mapSceneToView(
                    evt.scenePos()
                ).y()
            )
            if (
                (pos >= 0)
                and (
                    pos >= self.tabbed_area.contour_widget.view.getAxis("left").range[0]
                )
                and (
                    pos
                    < len(
                        self.settings.tiflist[
                            self.settings.keylist[self.settings.curr_key]
                        ]
                    )
                )
            ):
                if self.timer.isActive():
                    self.pause()
                self.settings.curr_pos = pos
                self.tabbed_area.contour_widget.horiz_line.setValue(pos)
                self.updateImages()

    def vLineCheckbox_changed(self, evt):
        if self.vLineCheckbox.isChecked():
            self.integral_widget.vLine.setPen(0, 0, 0, 150)
        else:
            self.integral_widget.vLine.setPen(0, 0, 0, 0)

    def circleCheckbox_changed(self, evt):
        if not self.circleCheckbox.isChecked():
            # self.imageview.tth_circle_RGBA[:,:,3] = 0
            self.imageview.tth_circle_data.set_opacity(0)
            self.imageview.tth_circle.updateImage(
                self.imageview.tth_circle_data.full_data
            )
        else:
            self.imageview.tth_circle_data.set_opacity(0.5)
            self.imageview.tth_circle.updateImage(
                self.imageview.tth_circle_data.full_data
            )

    def linked_axes_checkbox_changed(self, evt):
        if self.linked_axes_checkbox.isChecked():
            self.tabbed_area.contour_widget.view.setXLink(
                self.integral_widget.integral_view.getViewBox()
            )
        else:
            # self.tabbed_area.contour_widget.view.getViewBox().linkView(pg.ViewBox.XAxis,None)
            self.tabbed_area.contour_widget.view.setXLink(None)

    def x_axis_changed(self, evt):
        # Update integrals, contour graph
        # print(self.x_axis_choice.currentIndex())
        # print(evt)
        # self.integral_widget.change_x_axis_type(evt,self.settings.wavelength)
        # self.tabbed_area.contour_widget.change_x_axis_type(evt,self.settings.wavelength)
        self.integral_widget.change_x_axis_type(evt)
        self.tabbed_area.contour_widget.change_x_axis_type(evt)

    def live_view_image_checkbox_changed(self, evt):
        if self.live_view_image_checkbox.isChecked():
            self.play()
        else:
            self.pause()

    def play(self):
        # No need for a queue, just to periodically check for updates
        self.timer.timeout.connect(self.check_for_updates)
        self.timer.start(100)

    def pause(self):
        if self.live_view_image_checkbox.isChecked():
            self.live_view_image_checkbox.stateChanged.disconnect(
                self.live_view_image_checkbox_changed
            )
            self.live_view_image_checkbox.setChecked(False)
            self.live_view_image_checkbox.stateChanged.connect(
                self.live_view_image_checkbox_changed
            )
        self.timer.stop()


class image_mask:
    def __init__(self, shape: tuple, color: str):
        self.opacity = 1
        self.__shape = shape
        self.mask_data = np.zeros(shape, dtype=bool)
        self.full_data = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self.color = color
        self._update_color()

    def _update_color(self):
        colorRGB = pg.mkColor(self.color).getRgb()
        self.full_data[:, :, 0] = int(colorRGB[0])
        self.full_data[:, :, 1] = int(colorRGB[1])
        self.full_data[:, :, 2] = int(colorRGB[2])

    def set_color(self, color):
        self.color = color
        self._update_color()

    def set_shape(self, shape):
        self.__shape = shape
        self.mask_data = np.zeros(shape, dtype=bool)
        self.full_data = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self._update_color()

    def set_data(self, data):
        if data.shape != self.__shape:
            self.set_shape(data.shape)
        self.mask_data = data
        self.full_data[:, :, 3] = data * int(255 * self.opacity)

    def set_opacity(self, opacity):
        self.opacity = opacity
        self.full_data[:, :, 3] = self.mask_data * int(255 * self.opacity)


class MainImageView(pg.GraphicsLayoutWidget):
    def __init__(self, settings: Settings, parent=None):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__(parent)
        self.settings = settings
        self.view = self.addPlot(title="")
        self.view.setAspectLocked(True)
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)
        # images_exist = False
        # while not images_exist:
        #     images = glob.glob(self.directory+"/*.tif.metadata")
        #     if len(images) > 0:
        #         del images
        #         images_exist = True
        # del images_exist
        # #print(tiflist)
        # self.image_data = tf.imread(directory + "\\" + tiflist[keylist[curr_key]][curr_pos] + ".tif")
        # self.image = pg.ImageItem(self.image_data)
        self.image_data = np.zeros(self.settings.image_size)
        self.image = pg.ImageItem()

        self.view.addItem(self.image)
        self.intensityBar = pg.HistogramLUTItem()
        self.intensityBar.setImageItem(self.image)
        self.intensityBar.gradient.setColorMap(self.cmap)
        self.intensityBar.gradient.showTicks(show=False)
        self.addItem(self.intensityBar)
        # self.predef_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.predef_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.predef_mask_RGBA,"hotpink")
        self.predef_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["predef_mask"].color
        )
        self.nonpositive_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["nonpositive_mask"].color
        )
        # Call update mask data instead of reading these in
        # self.predef_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_predef.tif")
        # self.nonzero_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.nonzero_mask_vals = np.zeros(self.image_data.shape)
        # self.outlier_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.outlier_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.outlier_mask_RGBA,"green")
        self.outlier_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["outlier_mask"].color
        )  # green
        # self.nonzero_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_maskpolnonzero.tif")
        # TODO: nonzero version
        # self.nonzero_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_om.tif")
        self.predef_mask = pg.ImageItem(self.predef_mask_data.full_data, levels=None)
        # self.nonzero_mask = pg.ImageItem(self.nonzero_mask_RGBA,levels=None)
        self.outlier_mask = pg.ImageItem(self.outlier_mask_data.full_data, levels=None)
        # self.spot_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.arcs_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.spot_mask_vals = np.zeros(self.image_data.shape)
        # self.arcs_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.spot_mask_RGBA,"darkcyan")
        # self.update_mask_color(self.arcs_mask_RGBA,"maroon")
        self.spot_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["spot_mask"].color
        )  # darkcyan
        self.arcs_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["arcs_mask"].color
        )  # maroon
        self.spot_mask = pg.ImageItem(self.spot_mask_data.full_data, levels=None)
        self.arcs_mask = pg.ImageItem(self.arcs_mask_data.full_data, levels=None)
        # #closed masks
        # self.closed_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_spots_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_arcs_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_mask_vals = np.zeros(self.image_data.shape)
        # self.closed_spots_vals = np.zeros(self.image_data.shape)
        # self.closed_arcs_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.closed_mask_RGBA,"lime") #red
        # self.update_mask_color(self.closed_spots_RGBA,"cyan") #cyan
        # self.update_mask_color(self.closed_arcs_RGBA,"red") #pink
        # self.closed_mask = pg.ImageItem(self.closed_mask_RGBA,levels=None)
        # self.closed_spots = pg.ImageItem(self.closed_spots_RGBA,levels=None)
        # self.closed_arcs = pg.ImageItem(self.closed_arcs_RGBA,levels=None)

        self.masks = {
            self.predef_mask: [self.predef_mask_data, "_predef.tif"],
            # self.nonzero_mask: [self.nonzero_mask_RGBA,self.nonzero_mask_vals,"_om.tif"],
            self.outlier_mask: [self.outlier_mask_data, "_closedmask.tif"],
            self.spot_mask: [self.spot_mask_data, "_spots.tif"],
            self.arcs_mask: [self.arcs_mask_data, "_arcs.tif"],
            # self.closed_mask: [self.closed_mask_RGBA,self.closed_mask_vals,"_closedmask.tif"],
            # self.closed_spots: [self.closed_spots_RGBA,self.closed_spots_vals,"_spotsclosed.tif"],
            # self.closed_arcs: [self.closed_arcs_RGBA,self.closed_arcs_vals,"_arcsclosed.tif"],
        }

        self.mask_opacity_label = QtWidgets.QLabel("Mask Opacity:")
        self.mask_opacity_box = QtWidgets.QSpinBox()
        self.mask_opacity_box.setMinimum(0)
        self.mask_opacity_box.setMaximum(100)
        self.mask_opacity_box.setSingleStep(10)
        self.mask_opacity_box.setValue(100)
        self.mask_opacity_box.valueChanged.connect(self.mask_opacity_changed)

        # self.update_masks_data()

        # Initial draw places non-closed above closed so all visible
        # self.view.addItem(self.closed_mask)
        self.view.addItem(self.outlier_mask)
        # self.view.addItem(self.closed_spots)
        self.view.addItem(self.spot_mask)
        # self.view.addItem(self.closed_arcs)
        self.view.addItem(self.arcs_mask)
        self.view.addItem(self.predef_mask)

        self.predef_mask_box = QtWidgets.QCheckBox("Predefined Mask")
        self.mask_box = QtWidgets.QCheckBox("Outlier Mask")
        self.predef_mask_box.setChecked(True)
        self.mask_box.setChecked(True)
        self.predef_mask_box.stateChanged.connect(self.predef_box_changed)
        self.mask_box.stateChanged.connect(self.mask_box_changed)
        self.spot_mask_box = QtWidgets.QCheckBox("Spot Mask")
        self.arcs_mask_box = QtWidgets.QCheckBox("Texture Mask")
        self.spot_mask_box.setChecked(True)
        self.arcs_mask_box.setChecked(True)
        self.spot_mask_box.stateChanged.connect(self.spot_box_changed)
        self.arcs_mask_box.stateChanged.connect(self.arcs_box_changed)
        # self.closedmask_box = QtWidgets.QCheckBox("Closed Mask")
        # self.closedspots_box = QtWidgets.QCheckBox("Closed Spots Mask")
        # self.closedarcs_box = QtWidgets.QCheckBox("Closed Arcs Mask")
        # self.closedmask_box.setChecked(True)
        # self.closedspots_box.setChecked(True)
        # self.closedarcs_box.setChecked(True)
        # self.closedmask_box.stateChanged.connect(self.closedmask_box_changed)
        # self.closedspots_box.stateChanged.connect(self.closedspots_box_changed)
        # self.closedarcs_box.stateChanged.connect(self.closedarcs_box_changed)

        # one of the pyqtgraph examples shows the setDrawKernel() option for an image. Could make a zero mask image that gets modified by the draw kernel (brush; probably just want [[1.0]] but can resize) for a mask.

        # TODO: There might be more than one file here if they tried a previous experiment. Prompt for imctrl file name.
        # Wait until these files exist.
        # maps_loaded = False
        # while not maps_loaded:
        #     maps = glob.glob(self.settings.directory + "\\maps\\*.tif")
        #     if len(maps) > 0:
        #         time.sleep(1)
        #         del maps
        #         maps_loaded = True
        # del maps_loaded
        # self.tth_map = tf.imread(glob.glob(self.settings.directory+"\\maps\\*_2thetamap.tif")[0])
        # self.azim_map = tf.imread(glob.glob(self.settings.directory+"\\maps\\*_azmmap.tif")[0])
        self.tth_map = np.zeros(self.settings.image_size)
        self.azim_map = np.zeros(self.settings.image_size)

        # one option is to create a QGraphicsEllipseItem(x,y,width,height), but it would be better to use the 2th map
        # self.tth_circle_RGBA = np.zeros_like(self.spot_mask_RGBA)
        # self.update_mask_color(self.tth_circle_RGBA,"white")
        self.tth_circle_data = image_mask(
            self.settings.image_size, self.settings.colors["tth_circle_mask"].color
        )
        self.tth_circle_data.set_opacity(0.5)
        self.tth_circle = pg.ImageItem(self.tth_circle_data.full_data, levels=None)
        self.view.addItem(self.tth_circle)

    def update_dir(self):
        # Levels: z min and max
        # Range: x, y min and max
        # HistogramRange: visible axis range for z
        self.tth_map = tf.imread(
            glob.glob(os.path.join(self.settings.directory,"maps")+os.sep+"*_2thetamap.tif")[0]
        )
        self.azim_map = tf.imread(
            glob.glob(os.path.join(self.settings.directory,"maps")+os.sep+"*_azmmap.tif")[0]
        )
        self.predef_mask_data.set_color(self.settings.colors["predef_mask"].color)
        self.nonpositive_mask_data.set_color(
            self.settings.colors["nonpositive_mask"].color
        )
        self.outlier_mask_data.set_color(self.settings.colors["outlier_mask"].color)
        self.arcs_mask_data.set_color(self.settings.colors["arcs_mask"].color)
        self.spot_mask_data.set_color(self.settings.colors["spot_mask"].color)
        self.tth_circle_data.set_color(self.settings.colors["tth_circle_mask"].color)
        self.update_image_data(xy_reset=True, z_reset=True)
        self.update_masks_data()

    def update_image_data(self, xy_reset=False, z_reset=False):
        # global tiflist, keylist, curr_key, curr_pos
        self.image_data = tf.imread(
            os.path.join(
                self.settings.directory,
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + ".tif"
            )
        )
        if z_reset:
            maxval = np.percentile(self.image_data, 99.9)
            self.image.updateImage(
                self.image_data, autoRange=xy_reset, autoLevels=False
            )
            self.intensityBar.setLevels(min=0.0, max=maxval)
        else:
            self.image.updateImage(
                self.image_data,
                autoRange=xy_reset,
                autoLevels=z_reset,
                autoHistogramRange=z_reset,
            )

    # def update_mask_color(self,mask,color="red"):
    #     colorRGB = pg.mkColor(color).getRgb()
    #     mask[:,:,0] = int(colorRGB[0])
    #     mask[:,:,1] = int(colorRGB[1])
    #     mask[:,:,2] = int(colorRGB[2])

    def update_masks_data(self):
        # global tiflist, keylist, curr_key, curr_pos
        for mask,vals in self.masks.items():
            #print(tiflist[keylist[curr_key]][curr_pos])
            file_name = os.path.join(
                self.settings.directory,
                "masks",
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + vals[1]
            )
            #if os.path.exists(file_name):
            #    vals[0][:,:,3] = tf.imread(file_name)
            # else:
            #    vals[0][:,:,3] = 0
            # Handle cases where the file exists but is still being written or is otherwise corrupted
            try:
                # vals[1] = tf.imread(file_name)
                vals[0].set_data(tf.imread(file_name))
            except:
                vals[0].set_shape(self.settings.image_size)
            # vals[0][:,:,3] = vals[1]*int(255*self.mask_opacity_box.value()/100)
            mask.updateImage(vals[0].full_data)

    def update_tth_circle(self, tth, width=0.03):
        circle = (self.tth_map > tth - width) & (self.tth_map < tth + width)
        self.tth_circle_data.set_data(circle)
        self.tth_circle.updateImage(self.tth_circle_data.full_data)

    def mask_box_changed(self):
        if self.mask_box.isChecked():
            # self.view.addItem(self.outlier_mask)
            self.outlier_mask.setVisible(True)
        else:
            # self.view.removeItem(self.outlier_mask)
            self.outlier_mask.setVisible(False)

    def predef_box_changed(self):
        if self.predef_mask_box.isChecked():
            # self.view.addItem(self.predef_mask)
            self.predef_mask.setVisible(True)
        else:
            # self.view.removeItem(self.predef_mask)
            self.predef_mask.setVisible(False)

    def spot_box_changed(self):
        if self.spot_mask_box.isChecked():
            # self.view.addItem(self.spot_mask)
            self.spot_mask.setVisible(True)
        else:
            # self.view.removeItem(self.spot_mask)
            self.spot_mask.setVisible(False)

    def arcs_box_changed(self):
        if self.arcs_mask_box.isChecked():
            # self.view.addItem(self.arcs_mask)
            self.arcs_mask.setVisible(True)
        else:
            # self.view.removeItem(self.arcs_mask)
            self.arcs_mask.setVisible(False)

    # def closedmask_box_changed(self):
    #     if self.closedmask_box.isChecked():
    #         #self.view.addItem(self.closed_mask)
    #         self.closed_mask.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_mask)
    #         self.closed_mask.setVisible(False)
    # def closedspots_box_changed(self):
    #     if self.closedspots_box.isChecked():
    #         #self.view.addItem(self.closed_spots)
    #         self.closed_spots.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_spots)
    #         self.closed_spots.setVisible(False)
    # def closedarcs_box_changed(self):
    #     if self.closedarcs_box.isChecked():
    #         #self.view.addItem(self.closed_arcs)
    #         self.closed_arcs.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_arcs)
    #         self.closed_arcs.setVisible(False)

    def mask_opacity_changed(self, evt):
        for mask, vals in self.masks.items():
            # mask.setOpts(opacify=evt/100.)
            # vals[0][:,:,3] = vals[1]*int(255*evt/100)
            vals[0].set_opacity(evt / 100)
            mask.updateImage(vals[0].full_data)


class IntegralView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__()
        # self.setMaximumSize(800,300) # for demo
        self.setMinimumHeight(250)
        # self.directory = directory
        self.settings = settings
        self.setBackground("w")

        self.integral_offset_text = QtWidgets.QLabel()
        self.integral_offset_text.setText("Offset:")
        self.integral_offset = QtWidgets.QSpinBox()
        self.integral_offset.setMinimum(0)
        self.integral_offset.setMaximum(1000000)
        self.integral_offset.setSingleStep(100)
        self.integral_offset.setValue(0)
        self.integral_offset.valueChanged.connect(self.update_integral_offset)

        self.integral_view = self.addPlot(title="")
        self.axis_type = 0

        # integral_infile_piece = self.settings.directory + "\\integrals\\" + self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]
        # self.integral_data = np.loadtxt(integral_infile_piece+"_base.xye",skiprows=3)
        # self.masked_integral_data = np.loadtxt(integral_infile_piece+"_closed.xye",skiprows=3)
        # self.spotmasked_integral_data = np.loadtxt(integral_infile_piece+"_closedspotsmasked.xye",skiprows=3)
        # self.texturemasked_integral_data = np.loadtxt(integral_infile_piece+"_closedarcsmasked.xye",skiprows=3)
        self.integral_data = np.empty((self.settings.outChannels, 3))
        self.masked_integral_data = np.empty((self.settings.outChannels, 3))
        self.spotmasked_integral_data = np.empty((self.settings.outChannels, 3))
        self.texturemasked_integral_data = np.empty((self.settings.outChannels, 3))

        # self.base_integral = self.integral_view.plot(self.integral_data[:,0],self.integral_data[:,1],pen="black")
        # self.masked_integral = self.integral_view.plot(self.masked_integral_data[:,0],self.masked_integral_data[:,1] + self.integral_offset.value(),pen="green")
        # self.spotmasked_integral = self.integral_view.plot(self.spotmasked_integral_data[:,0],self.spotmasked_integral_data[:,1] + 2*self.integral_offset.value(),pen="darkcyan")
        # self.texturemasked_integral = self.integral_view.plot(self.texturemasked_integral_data[:,0],self.texturemasked_integral_data[:,1] + 3*self.integral_offset.value(),pen="maroon")
        # self.spots_diff_integral = self.integral_view.plot(self.spotmasked_integral_data[:,0],self.integral_data[:,1]-self.spotmasked_integral_data[:,1],pen=(0,0,0,0)) #draw with an invisible pen to start off
        # self.arcs_diff_integral = self.integral_view.plot(self.texturemasked_integral_data[:,0],self.integral_data[:,1]-self.texturemasked_integral_data[:,1],pen=(0,0,0,0))
        self.base_integral = self.integral_view.plot(
            pen=self.settings.colors["base_line"].color
        )
        self.masked_integral = self.integral_view.plot(
            pen=self.settings.colors["outlier_line"].color
        )
        self.spotmasked_integral = self.integral_view.plot(
            pen=self.settings.colors["spot_line"].color
        )
        self.texturemasked_integral = self.integral_view.plot(
            pen=self.settings.colors["arcs_line"].color
        )
        self.spots_diff_integral = self.integral_view.plot(
            pen=self.settings.colors["minus_spot_line"].color
        )
        self.arcs_diff_integral = self.integral_view.plot(
            pen=self.settings.colors["minus_arcs_line"].color
        )
        self.spots_diff_integral.setVisible(False)
        self.arcs_diff_integral.setVisible(False)

        # Vertical line showing mouse 2theta position from the main image
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.integral_view.addItem(self.vLine, ignoreBounds=True)

        # generalize the integral checkboxes
        # define a pen for each, have checkbox swap between (0,0,0,0) and its pen
        self.base_integral_checkbox = QtWidgets.QCheckBox("Base Integral")
        self.base_integral_checkbox.setChecked(True)
        self.base_integral_checkbox.stateChanged.connect(
            self.base_integral_checkbox_changed
        )

        self.masked_integral_checkbox = QtWidgets.QCheckBox("Outlier Masked")
        self.masked_integral_checkbox.setChecked(True)
        self.masked_integral_checkbox.stateChanged.connect(
            self.masked_integral_checkbox_changed
        )

        self.spotmasked_integral_checkbox = QtWidgets.QCheckBox("Spot Masked")
        self.spotmasked_integral_checkbox.setChecked(True)
        self.spotmasked_integral_checkbox.stateChanged.connect(
            self.spotmasked_integral_checkbox_changed
        )

        self.texturemasked_integral_checkbox = QtWidgets.QCheckBox("Texture Masked")
        self.texturemasked_integral_checkbox.setChecked(True)
        # colorCount, palette, style, repaint
        # self.texturemasked_integral_checkbox.setPalette()
        self.texturemasked_integral_checkbox.stateChanged.connect(
            self.texturemasked_integral_checkbox_changed
        )

        self.spots_diff_integral_checkbox = QtWidgets.QCheckBox(
            "Base \u2013 Spot Masked"
        )
        self.spots_diff_integral_checkbox.stateChanged.connect(
            self.spots_diff_integral_checkbox_changed
        )
        self.arcs_diff_integral_checkbox = QtWidgets.QCheckBox(
            "Base \u2013 Texture Masked"
        )
        self.arcs_diff_integral_checkbox.stateChanged.connect(
            self.arcs_diff_integral_checkbox_changed
        )

        self.sqrt_checkbox = QtWidgets.QCheckBox("Sqrt(y)")
        self.sqrt_checkbox.stateChanged.connect(self.sqrt_toggle)

        # set an order
        # or set the text color to a faded gray when it's toggled off to show it can be there and still keep the order
        self.legend = self.integral_view.addLegend(offset=(-1, 1))
        self.legend.addItem(self.base_integral, "Base Integral")
        self.legend.addItem(self.masked_integral, "Outlier Masked")
        self.legend.addItem(self.spotmasked_integral, "Spot Masked")
        self.legend.addItem(self.texturemasked_integral, "Texture Masked")
        self.legend.addItem(self.spots_diff_integral, "Base \u2013 Spot Masked")
        self.legend.addItem(self.arcs_diff_integral, "Base \u2013 Texture Masked")

    def update_dir(self):
        # print("Integrals: updating directory")
        self.integral_data = np.empty((self.settings.outChannels, 3))
        self.masked_integral_data = np.empty((self.settings.outChannels, 3))
        self.spotmasked_integral_data = np.empty((self.settings.outChannels, 3))
        self.texturemasked_integral_data = np.empty((self.settings.outChannels, 3))
        self.base_integral.setPen(self.settings.colors["base_line"].color)
        self.masked_integral.setPen(self.settings.colors["outlier_line"].color)
        self.spotmasked_integral.setPen(self.settings.colors["spot_line"].color)
        self.texturemasked_integral.setPen(self.settings.colors["arcs_line"].color)
        self.arcs_diff_integral.setPen(self.settings.colors["minus_arcs_line"].color)
        self.spots_diff_integral.setPen(self.settings.colors["minus_spot_line"].color)
        self.update_integral_data()

    def update_integral_data(self):
        # print("Integrals: updating data")
        # global tiflist, keylist, curr_key, curr_pos
        integral_infile_piece = os.path.join(
            self.settings.directory,
            "integrals",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]
        )
        integrals_dict = {
            "_base.xye": self.integral_data,
            "_closed.xye": self.masked_integral_data,
            "_closedspotsmasked.xye": self.spotmasked_integral_data,
            "_closedarcsmasked.xye": self.texturemasked_integral_data,
            # "_closed.xye": self.closed_integral_data,
            # "_closedarcsmasked.xye": self.closedarcs_integral_data,
            # "_closedspotsmasked.xye": self.closedspots_integral_data,
            # "_closed_pytorch.xye": self.pytorch_integral_data,
        }
        for ext, vals in integrals_dict.items():
            # print("Integrals: loading data for {0}".format(ext))
            try:
                new_vals = np.loadtxt(integral_infile_piece + ext, skiprows=3)
            except:
                print(
                    "Exception reading in integral data for {0}. Setting to 0.".format(
                        integral_infile_piece + ext
                    )
                )
                vals[:, :] = 0
            if new_vals.shape != vals.shape:
                print(
                    "Shape of incoming data {0} not equal to config shape {1} for integral {2}. Setting values to 0. If the config shape looks incorrect, go to Settings->outChannels and set it to {3}.".format(
                        new_vals.shape,
                        vals.shape,
                        integral_infile_piece + ext,
                        new_vals.shape[0],
                    )
                )
                vals[:, :] = 0
            try:
                vals[:, :] = new_vals
            except:
                print(
                    "Exception setting integral data for {0}. Setting to 0.".format(
                        integral_infile_piece, ext
                    )
                )

        if self.axis_type == 0:
            self.base_integral.setData(
                self.integral_data[:, 0], self.integral_data[:, 1]
            )
            self.masked_integral.setData(
                self.masked_integral_data[:, 0],
                self.masked_integral_data[:, 1] + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                self.spotmasked_integral_data[:, 0],
                self.spotmasked_integral_data[:, 1] + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                self.texturemasked_integral_data[:, 0],
                self.texturemasked_integral_data[:, 1]
                + 3 * self.integral_offset.value(),
            )
            self.spots_diff_integral.setData(
                self.spotmasked_integral_data[:, 0],
                self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1],
            )
            self.arcs_diff_integral.setData(
                self.texturemasked_integral_data[:, 0],
                self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1],
            )

        elif self.axis_type == 1:
            self.base_integral.setData(
                tth_to_q(self.integral_data[:, 0], self.settings.wavelength),
                self.integral_data[:, 1],
            )
            self.masked_integral.setData(
                tth_to_q(self.masked_integral_data[:, 0], self.settings.wavelength),
                self.masked_integral_data[:, 1] + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                tth_to_q(self.spotmasked_integral_data[:, 0], self.settings.wavelength),
                self.spotmasked_integral_data[:, 1] + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                tth_to_q(
                    self.texturemasked_integral_data[:, 0], self.settings.wavelength
                ),
                self.texturemasked_integral_data[:, 1]
                + 3 * self.integral_offset.value(),
            )
            self.spots_diff_integral.setData(
                tth_to_q(self.spotmasked_integral_data[:, 0], self.settings.wavelength),
                self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1],
            )
            self.arcs_diff_integral.setData(
                tth_to_q(
                    self.texturemasked_integral_data[:, 0], self.settings.wavelength
                ),
                self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1],
            )

    def sqrt_toggle(self, evt):
        if self.sqrt_checkbox.isChecked():
            # Toggle offset
            # self.integral_offset.valueChanged.disconnect()
            # self.integral_offset.setValue(int(self.integral_offset.value()/10))
            self.integral_offset.setValue(
                int(np.round(np.sqrt(self.integral_offset.value()), decimals=-1))
            )
            # self.integral_offset.valueChanged.connect(self.update_integral_offset)
            self.integral_offset.setSingleStep(10)
            # Toggle data. Diff integrals have some negative values.
            self.base_integral.setData(
                self.integral_data[:, 0], np.sqrt(self.integral_data[:, 1])
            )
            self.masked_integral.setData(
                self.masked_integral_data[:, 0],
                np.sqrt(self.masked_integral_data[:, 1]) + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                self.spotmasked_integral_data[:, 0],
                np.sqrt(self.spotmasked_integral_data[:, 1])
                + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                self.texturemasked_integral_data[:, 0],
                np.sqrt(self.texturemasked_integral_data[:, 1])
                + 3 * self.integral_offset.value(),
            )
            # self.spots_diff_integral.setData(self.spotmasked_integral_data[:,0],np.sqrt(self.integral_data[:,1]-self.spotmasked_integral_data[:,1]))
            # self.arcs_diff_integral.setData(self.texturemasked_integral_data[:,0],np.sqrt(self.integral_data[:,1]-self.texturemasked_integral_data[:,1]))

        else:
            # Toggle offset
            # self.integral_offset.setValue(self.integral_offset.value()*10)
            self.integral_offset.setValue(
                int(np.round(self.integral_offset.value() ** 2, decimals=-2))
            )
            self.integral_offset.setSingleStep(100)
            # Toggle data. Diff integrals have some negative values.
            self.base_integral.setData(
                self.integral_data[:, 0], self.integral_data[:, 1]
            )
            self.masked_integral.setData(
                self.masked_integral_data[:, 0],
                self.masked_integral_data[:, 1] + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                self.spotmasked_integral_data[:, 0],
                self.spotmasked_integral_data[:, 1] + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                self.texturemasked_integral_data[:, 0],
                self.texturemasked_integral_data[:, 1]
                + 3 * self.integral_offset.value(),
            )
            # self.spots_diff_integral.setData(self.spotmasked_integral_data[:,0],self.integral_data[:,1]-self.spotmasked_integral_data[:,1])
            # self.arcs_diff_integral.setData(self.texturemasked_integral_data[:,0],self.integral_data[:,1]-self.texturemasked_integral_data[:,1])

    def change_x_axis_type(self, axis_type):
        # Will be passed index of axis type
        # 2 theta = 0, Q = 1
        self.axis_type = axis_type
        # self.wavelength = wavelength
        self.update_integral_data()

    def update_integral_offset(self):
        if self.sqrt_checkbox.isChecked():
            self.masked_integral.setData(
                self.masked_integral_data[:, 0],
                np.sqrt(self.masked_integral_data[:, 1]) + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                self.spotmasked_integral_data[:, 0],
                np.sqrt(self.spotmasked_integral_data[:, 1])
                + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                self.texturemasked_integral_data[:, 0],
                np.sqrt(self.texturemasked_integral_data[:, 1])
                + 3 * self.integral_offset.value(),
            )
            # self.spots_diff_integral.setData(self.spots_diff_data[:,0],self.spots_diff_data[:,1])
            # self.arcs_diff_integral.setData(self.arcs_diff_data[:,0],self.arcs_diff_data[:,1])
            # self.closed_integral.setData(self.closed_integral_data[:,0],np.sqrt(self.closed_integral_data[:,1]) + self.integral_offset.value())
            # self.closedspots_integral.setData(self.closedspots_integral_data[:,0],np.sqrt(self.closedspots_integral_data[:,1]) + 2*self.integral_offset.value())
            # self.closedarcs_integral.setData(self.closedarcs_integral_data[:,0],np.sqrt(self.closedarcs_integral_data[:,1]) + 3*self.integral_offset.value())
        else:
            self.masked_integral.setData(
                self.masked_integral_data[:, 0],
                self.masked_integral_data[:, 1] + self.integral_offset.value(),
            )
            self.spotmasked_integral.setData(
                self.spotmasked_integral_data[:, 0],
                self.spotmasked_integral_data[:, 1] + 2 * self.integral_offset.value(),
            )
            self.texturemasked_integral.setData(
                self.texturemasked_integral_data[:, 0],
                self.texturemasked_integral_data[:, 1]
                + 3 * self.integral_offset.value(),
            )
            # self.spots_diff_integral.setData(self.spots_diff_data[:,0],self.spots_diff_data[:,1])
            # self.arcs_diff_integral.setData(self.arcs_diff_data[:,0],self.arcs_diff_data[:,1])
            # self.closed_integral.setData(self.closed_integral_data[:,0],self.closed_integral_data[:,1] + self.integral_offset.value())
            # self.closedspots_integral.setData(self.closedspots_integral_data[:,0],self.closedspots_integral_data[:,1] + 2*self.integral_offset.value())
            # self.closedarcs_integral.setData(self.closedarcs_integral_data[:,0],self.closedarcs_integral_data[:,1] + 3*self.integral_offset.value())

    def base_integral_checkbox_changed(self):
        if self.base_integral_checkbox.isChecked():
            # self.base_integral.setPen(("black"))
            self.base_integral.setVisible(True)
        else:
            # self.base_integral.setPen((255,255,255,0))
            self.base_integral.setVisible(False)

    def masked_integral_checkbox_changed(self):
        if self.masked_integral_checkbox.isChecked():
            # self.masked_integral.setPen("green")
            self.masked_integral.setVisible(True)
        else:
            # self.masked_integral.setPen((200,100,0,0))
            self.masked_integral.setVisible(False)

    def spotmasked_integral_checkbox_changed(self):
        if self.spotmasked_integral_checkbox.isChecked():
            # self.spotmasked_integral.setPen("darkcyan")
            self.spotmasked_integral.setVisible(True)
        else:
            # self.spotmasked_integral.setPen((0,255,0,0))
            self.spotmasked_integral.setVisible(False)

    def texturemasked_integral_checkbox_changed(self):
        if self.texturemasked_integral_checkbox.isChecked():
            # self.texturemasked_integral.setPen("maroon")
            self.texturemasked_integral.setVisible(True)
        else:
            # self.texturemasked_integral.setPen((0,0,255,0))
            self.texturemasked_integral.setVisible(False)

    def spots_diff_integral_checkbox_changed(self):
        if self.spots_diff_integral_checkbox.isChecked():
            # self.spots_diff_integral.setPen(self.settings.colors["minus_spot_line"].color)
            self.spots_diff_integral.setVisible(True)
        else:
            # self.spots_diff_integral.setPen(0,0,0,0)
            self.spots_diff_integral.setVisible(False)

    def arcs_diff_integral_checkbox_changed(self):
        if self.arcs_diff_integral_checkbox.isChecked():
            # self.arcs_diff_integral.setPen(self.settings.colors["minus_arcs_line"].color)
            self.arcs_diff_integral.setVisible(True)
        else:
            # self.arcs_diff_integral.setPen(0,0,0,0)
            self.arcs_diff_integral.setVisible(False)

    # def closedmasked_integral_checkbox_changed(self):
    #     if self.closedmasked_integral_checkbox.isChecked():
    #         #self.closed_integral.setPen("lime")
    #         self.closed_integral.setVisible(True)
    #     else:
    #         #self.closed_integral.setPen((200,100,0,0))
    #         self.closed_integral.setVisible(False)

    # def closedspots_integral_checkbox_changed(self):
    #     if self.closedspots_integral_checkbox.isChecked():
    #         #self.closedspots_integral.setPen("cyan")
    #         self.closedspots_integral.setVisible(True)
    #     else:
    #         #self.closedspots_integral.setPen((0,255,0,0))
    #         self.closedspots_integral.setVisible(False)

    # def closedarcs_integral_checkbox_changed(self):
    #     if self.closedarcs_integral_checkbox.isChecked():
    #         #self.closedarcs_integral.setPen("red")
    #         self.closedarcs_integral.setVisible(True)
    #     else:
    #         #self.closedarcs_integral.setPen((0,0,255,0))
    #         self.closedarcs_integral.setVisible(False)

    # def closedspots_diff_integral_checkbox_changed(self):
    #     if self.closedspots_diff_integral_checkbox.isChecked():
    #         self.closedspots_diff_integral.setPen('k')
    #     else:
    #         self.closedspots_diff_integral.setPen(0,0,0,0)

    # def closedarcs_diff_integral_checkbox_changed(self):
    #     if self.closedarcs_diff_integral_checkbox.isChecked():
    #         self.closedarcs_diff_integral.setPen('k')
    #     else:
    #         self.closedarcs_diff_integral.setPen(0,0,0,0)


class StatsView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__(parent)
        # self.directory = directory
        self.settings = settings
        self.setBackground("w")
        self.stats_view = self.addPlot(title="")
        self.spots_stats_hist = None
        self.spots_histogram_area = None
        self.spots_histogram_Q = None
        self.x_bins = None
        self.y_bins = None

        self.spots_histogram_area_curve = pg.PlotCurveItem(
            fillLevel=0, brush=(0, 0, 255, 80)
        )
        self.spots_histogram_Q_curve = pg.PlotCurveItem(
            fillLevel=0, brush=(0, 255, 0, 80)
        )
        self.spots_histogram_area_Q = pg.ImageItem()

        self.stats_view.addItem(self.spots_histogram_area_curve)
        # self.stats_view.addItem(self.spots_histogram_Q_curve)
        self.legend = self.stats_view.addLegend(offset=(-1, 1))
        self.legend.addItem(self.spots_histogram_area_curve, "Spot Area")
        # self.legend.addItem(self.spots_histogram_Q_curve,"Q Position of Spots")

        # UI
        self.histogram_type_select = QtWidgets.QComboBox()
        self.histogram_type_dict = {
            "Spot Area": self.spots_histogram_area_curve,
            "Spot Q Position": self.spots_histogram_Q_curve,
            "Area vs Q": self.spots_histogram_area_Q,
        }
        self.histogram_types = list(self.histogram_type_dict.keys())
        print(self.histogram_types)
        self.histogram_type_select.addItems(self.histogram_types)
        self.histogram_type_select.setCurrentIndex(0)
        self.histogram_type_select.currentIndexChanged.connect(
            self.histogram_type_changed
        )

    def histogram_type_changed(self, evt):
        self.stats_view.clear()
        self.legend.clear()
        self.stats_view.addItem(self.histogram_type_dict[self.histogram_types[evt]])
        if evt != 2:
            self.legend.addItem(
                self.histogram_type_dict[self.histogram_types[evt]],
                self.histogram_types[evt],
            )

    def update_stats_data(self):
        stats_infile = os.path.join(
            self.settings.directory,
            "stats",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_spots_hist.npy"
        )
        with open(stats_infile, 'rb') as infile:
            self.spots_stats_hist = np.load(infile)
            self.area_bins = np.load(infile)
            self.Q_bins = np.load(infile)
        self.spots_histogram_area = np.sum(self.spots_stats_hist, axis=1)
        self.spots_histogram_Q = np.sum(self.spots_stats_hist, axis=0)
        self.spots_histogram_area_curve.setData(
            self.area_bins, self.spots_histogram_area, stepMode="center"
        )
        self.spots_histogram_Q_curve.setData(
            self.Q_bins, self.spots_histogram_Q, stepMode="center"
        )
        self.spots_histogram_area_Q.setImage(self.spots_stats_hist)

    # def update_stats_data(self):
    #     global tiflist, keylist, curr_key, curr_pos
    #     # stats_infile = "integrals\\" + tiflist[keylist[curr_key]][curr_pos] + "_stats.npy"
    #     spots_stats_infile = self.directory + "\\stats\\" + tiflist[keylist[curr_key]][curr_pos] + "_spots_stats.csv"
    #     arcs_stats_infile = self.directory + "\\stats\\" + tiflist[keylist[curr_key]][curr_pos] + "_arcs_stats.csv"
    #     # print(spots_stats_infile)
    #     spots_stats = pd.read_csv(spots_stats_infile)
    #     arcs_stats = pd.read_csv(arcs_stats_infile)
    #     y,x = np.histogram(spots_stats['area'].values,bins=200)
    #     self.spots_histogram_curve.setData(x,y,stepMode="center")
    #     y,x = np.histogram(arcs_stats['area'].values,bins=200)
    #     self.arcs_histogram_curve.setData(x,y,stepMode="center")

    def update_dir(self):
        self.update_stats_data()
        return


class CSimView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        # global tiflist, keylist, curr_key, curr_pos
        self.setBackground("w")
        # self.directory = directory
        self.settings = settings
        # similar to contour view, show full directory
        self.view = self.addPlot(title="")
        self.min = 0
        self.max = 100
        self.spacing = 1
        # self.methods = ["Cosine Similarity", "Normalized Mutual Information", "Structural Similarity"]
        # self.methods_ext = ["_csim.csv", "_nmi.csv", "_ssim.csv"]
        self.methods = {
            "Cosine Similarity": "_csim.txt",
            # "Normalized Mutual Information": "_nmi.txt",
            # "Structural Similarity": "_ssim.txt",
        }
        self.similarity_line = {}
        self.similarity_line_data = {}
        self.legend = self.view.addLegend(offset=(-1, 1))
        for k, v in self.methods.items():
            # check if addPlot.plot() is a shortcut for create PlotItem, add PlotItem
            self.similarity_line[k] = self.view.plot()
            self.similarity_line_data[k] = []
            self.legend.addItem(self.similarity_line[k], k)
        self.similarity_line["Cosine Similarity"].setPen("r")
        # self.similarity_line["Normalized Mutual Information"].setPen('g')
        # self.similarity_line["Structural Similarity"].setPen('b')

    def update_dir(self):
        self.update_data()

    def update_data(self):
        # global keylist, curr_key
        for k, v in self.methods.items():
            # Two files get created, one with 00000 and one without. Try finding the former first; latter only appears if there is only one file.
            filename0 = os.path.join(
                self.settings.directory,
                "stats",
                self.settings.keylist[self.settings.curr_key] + "00000" + v
            )
            filename  = os.path.join(
                self.settings.directory,
                "stats",
                self.settings.keylist[self.settings.curr_key][:-1] + v
            ) # testing shows the - at the end, need to remove
            if os.path.exists(filename0):
                self.similarity_line_data[k] = np.loadtxt(filename0, skiprows=2)
            elif os.path.exists(filename):
                self.similarity_line_data[k] = np.loadtxt(filename, skiprows=2)
            # [:,0] for comparison to first, [:,1] for comparison to previous
            self.similarity_line[k].setData(self.similarity_line_data[k][:, 0])


class ContourView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.setMinimumHeight(150)
        # self.directory = directory
        self.settings = settings
        self.live_max_lines_visible = 100
        self.live_spacing = 1
        self._temp_auto_spacing = 1
        self.live_min = 0
        self.live_min_changed = False
        self.live_step_changed = False
        self.live_max_changed = False
        self.manual_min = 0
        self.manual_max = 100
        self.manual_spacing = 1
        self.integral_extension = "_closed.xye"
        self.integral_data = []
        self.automatically_set_spacing = True
        # Intended to be "Should this zoom out (change step size) when more images appear than the max, or should it start scrolling?" Not yet implemented.
        self.modes = ["zoom", "scroll"]
        self.mode = "zoom"
        # xvals will swap between tthvals and qvals
        self.x_axis_type = 0  # 0 = tth, 1 = Q
        self.tthvals = []
        self.qvals = []
        self.yvals = []
        # self.wavelength = self.settings.wavelength
        self.timer = QtCore.QTimer()

        self.view = self.addPlot(title="")
        self.view.setAspectLocked(False)
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)
        # self.image_data = tf.imread(tiflist[keylist[curr_key]][curr_pos] + ".tif")
        self.contour_image = pg.ImageItem()
        self.view.addItem(self.contour_image)
        self.intensityBar = pg.HistogramLUTItem()
        self.intensityBar.setImageItem(self.contour_image)
        self.intensityBar.gradient.setColorMap(self.cmap)
        self.intensityBar.gradient.showTicks(show=False)
        self.addItem(self.intensityBar)

        self.tth_line = pg.InfiniteLine(angle=90, movable=False)
        self.view.addItem(self.tth_line, ignoreBounds=True)
        self.horiz_line = pg.InfiniteLine(angle=0, movable=False)
        self.view.addItem(self.horiz_line, ignoreBounds=True)

        # self.update_integral_list()

        self.live_update_checkbox = QtWidgets.QCheckBox("Live update")
        self.live_update_checkbox.stateChanged.connect(
            self.live_update_checkbox_changed
        )
        self.tth_line_checkbox = QtWidgets.QCheckBox("2Th Line")
        self.tth_line_checkbox.stateChanged.connect(self.tth_line_checkbox_changed)

        self.integral_select = QtWidgets.QComboBox()
        # self.integral_types = ["Base","Outlier Masked","Closed Mask"]
        self.integral_type_dict = {
            "Base": "_base.xye",
            # "Outlier Masked":"_om.xye",
            "Closed Mask": "_closed.xye",
        }
        self.integral_types = list(self.integral_type_dict.keys())
        self.integral_select.addItems(self.integral_types)
        self.integral_select.setCurrentIndex(1)
        self.integral_select.currentIndexChanged.connect(self.integral_type_changed)

        self.live_controls_list = []
        self.live_integral_min_label = QtWidgets.QLabel()
        self.live_integral_min_label.setText("Min:")
        self.live_controls_list.append(self.live_integral_min_label)
        self.live_integral_min = QtWidgets.QSpinBox()
        self.live_integral_min.setMinimum(0)
        self.live_integral_min.valueChanged.connect(self.live_integral_min_changed)
        self.live_controls_list.append(self.live_integral_min)
        self.live_integral_max_label = QtWidgets.QLabel()
        self.live_integral_max_label.setText("Max integrals drawn:")
        self.live_controls_list.append(self.live_integral_max_label)
        self.live_integral_max = QtWidgets.QSpinBox()
        self.live_integral_max.setMinimum(1)
        self.live_integral_max.setMaximum(10000)
        self.live_integral_max.setValue(100)
        self.live_integral_max.valueChanged.connect(self.live_integral_max_changed)
        self.live_controls_list.append(self.live_integral_max)
        self.live_integral_step_label = QtWidgets.QLabel()
        self.live_integral_step_label.setText("Step size:")
        self.live_controls_list.append(self.live_integral_step_label)
        self.live_integral_step = QtWidgets.QSpinBox()
        self.live_integral_step.setMinimum(1)
        self.live_integral_step.valueChanged.connect(self.live_integral_step_changed)
        self.live_controls_list.append(self.live_integral_step)

        for widget in self.live_controls_list:
            widget.setEnabled(False)

        self.manual_controls_list = []
        self.integral_min_label = QtWidgets.QLabel()
        self.integral_min_label.setText("Min:")
        self.manual_controls_list.append(self.integral_min_label)
        self.integral_min = QtWidgets.QSpinBox()
        self.integral_min.setMinimum(0)
        self.integral_min.valueChanged.connect(self.manual_integral_min_changed)
        self.manual_controls_list.append(self.integral_min)
        self.integral_max_label = QtWidgets.QLabel()
        self.integral_max_label.setText("Max:")
        self.manual_controls_list.append(self.integral_max_label)
        self.integral_max = QtWidgets.QSpinBox()
        self.integral_max.setMinimum(1)
        self.integral_max.setMaximum(10000)
        self.integral_max.setValue(100)
        self.integral_max.valueChanged.connect(self.manual_integral_max_changed)
        self.manual_controls_list.append(self.integral_max)
        self.integral_step_label = QtWidgets.QLabel()
        self.integral_step_label.setText("Step size:")
        self.manual_controls_list.append(self.integral_step_label)
        self.integral_step = QtWidgets.QSpinBox()
        self.integral_step.setMinimum(1)
        self.integral_step.valueChanged.connect(self.manual_integral_step_changed)
        self.manual_controls_list.append(self.integral_step)

    def update_dir(self):
        self.reset_integral_data()

    def update_integral_list(self):
        # global keylist, curr_key
        self.integral_filelist = sorted(
            glob.glob(
                os.path.join(
                    self.settings.directory,
                    "integrals",
                    self.settings.keylist[self.settings.curr_key]
                )
                + os.sep
                + "*"
                + self.integral_extension
            ),
            key=os.path.getctime
        )
        #Pop the last element of the list if it's been created in the past half second to avoid reading it while it is written
        #Test files showing 0.02 seconds from creation time to modification time
        #print(os.path.getmtime(self.integral_filelist[-1]) - os.path.getctime(self.integral_filelist[-1]))
        if (len(self.integral_filelist) > 0) and (
            time.time() - os.path.getctime(self.integral_filelist[-1]) < 0.5
        ):
            self.integral_filelist.pop()
        if self.automatically_set_spacing:
            self.auto_set_spacing()
        self.update_integral_data()

    def auto_set_spacing(self):
        # while len(self.integral_filelist) >= (self.max_lines_visible + 1)*self.requested_spacing:
        #    self.requested_spacing *= 2
        self._temp_auto_spacing = self.live_spacing
        while len(self.integral_filelist[self.live_min :: self._temp_auto_spacing]) >= (
            self.live_max_lines_visible + 1
        ):
            self._temp_auto_spacing *= 2

    def append_integral_data(self, max=None):
        # append data from files beyond cur_len*spacing
        cur_len = len(self.integral_data)
        if max == None:
            file_subset = self.integral_filelist[self.live_min :: self.live_spacing]
        else:
            file_subset = self.integral_filelist[
                self.live_min : max : self.live_spacing
            ]
        for i in range(cur_len, len(file_subset)):
            data = np.loadtxt(file_subset[i], skiprows=3)
            if len(self.tthvals) == 0:
                self.tthvals = data[:, 0]
                if self.settings.wavelength != 0:
                    self.qvals = tth_to_q(self.tthvals, self.settings.wavelength)
                # self.surface_plot_item.setData(x=np.array(self.xvals))
                # self.surface_plot_item.shader()['colorMap'] = pyqt_default_shader_scaled(np.array(data[:,1]))
            self.yvals.append((len(self.yvals)) * self.live_spacing + self.live_min)
            self.integral_data.append(data[:, 1])
        # print(np.array(self.xvals).shape, np.array(self.yvals).shape,np.transpose(self.integral_data))
        self.contour_image.setImage(np.array(self.integral_data))
        if self.x_axis_type == 0:
            self.contour_image.setRect(
                self.tthvals[0],
                self.yvals[0],
                self.tthvals[-1] - self.tthvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )
        elif self.x_axis_type == 1:
            self.contour_image.setRect(
                self.qvals[0],
                self.yvals[0],
                self.qvals[-1] - self.qvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )

    def update_integral_data(self):
        # If the requested spacing hasn't changed, just append the new data
        if self._temp_auto_spacing == self.live_spacing:
            self.append_integral_data()
        else:
            # If requested spacing is divisible by old spacing, can just splice down and add back
            if self._temp_auto_spacing % self.live_spacing == 0:
                self.integral_data = self.integral_data[
                    0 :: int(self._temp_auto_spacing / self.live_spacing)
                ]
                self.yvals = self.yvals[
                    0 :: int(self._temp_auto_spacing / self.live_spacing)
                ]
                self.live_spacing = self._temp_auto_spacing
                self.live_integral_step.valueChanged.disconnect(
                    self.live_integral_step_changed
                )
                self.live_integral_step.setValue(self.live_spacing)
                self.live_integral_step.valueChanged.connect(
                    self.live_integral_step_changed
                )
                self.append_integral_data()
            # Otherwise, remake the dataset
            else:
                self.integral_data = []
                self.yvals = []
                self.live_spacing = self._temp_auto_spacing
                self.live_integral_step.valueChanged.disconnect(
                    self.live_integral_step_changed
                )
                self.live_integral_step.setValue(self.live_spacing)
                self.live_integral_step.valueChanged.connect(
                    self.live_integral_step_changed
                )
                self.append_integral_data()

    def change_x_axis_type(self, axis_type):
        # 2 theta = 0, Q = 1
        self.axis_type = axis_type
        # self.wavelength = wavelength
        if axis_type == 0:
            self.contour_image.setRect(
                self.tthvals[0],
                self.yvals[0],
                self.tthvals[-1] - self.tthvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )
        elif axis_type == 1:
            if len(self.qvals) == 0:
                self.qvals = tth_to_q(self.tthvals, self.settings.wavelength)
            self.contour_image.setRect(
                self.qvals[0],
                self.yvals[0],
                self.qvals[-1] - self.qvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )

    def update_integral_data_manual(self):
        self.live_spacing = self.integral_step.value()
        self.append_integral_data(max=self.manual_max + 1)
        # if self._temp_auto_spacing == self.live_spacing:
        #    self.append_integral_data(max=self.manual_max)
        # else:
        #    if self._temp_auto_spacing % self.live_spacing == 0:
        #        self.integral_data = self.integral_data[]

    def reset_integral_data(self, manual=False):
        self.integral_data = []
        self.xvals = []
        self.yvals = []
        if manual:
            self.update_integral_data_manual()
        else:
            self.live_spacing = 1
            self._temp_auto_spacing = 1
            self.live_integral_step.valueChanged.disconnect(
                self.live_integral_step_changed
            )
            self.live_integral_step.setValue(self.live_spacing)
            self.live_integral_step.valueChanged.connect(
                self.live_integral_step_changed
            )
            self.update_integral_list()

    def integral_type_changed(self, evt):
        self.integral_extension = self.integral_type_dict[self.integral_types[evt]]
        # TODO: only reset list/data; keep settings
        self.reset_integral_data()

    def update_integral_live(self):
        reset = False
        if self.live_min_changed:
            self.live_min = self.live_integral_min.value()
            self.live_min_changed = False
            reset = True
        if self.live_step_changed:
            self._temp_auto_spacing = self.live_integral_step.value()
            self.live_step_changed = False
        if self.live_max_changed:
            self.live_max_lines_visible = self.live_integral_max.value()
            self.live_max_changed = False
            reset = True
        if reset:
            self.reset_integral_data()
        else:
            self.update_integral_list()

    def pause(self):
        self.timer.stop()
        for widget in self.live_controls_list:
            widget.setEnabled(False)
        for widget in self.manual_controls_list:
            widget.setEnabled(True)
        self.live_min = self.integral_min.value()
        self.reset_integral_data(manual=True)

    def play(self):
        for widget in self.manual_controls_list:
            widget.setEnabled(False)
        for widget in self.live_controls_list:
            widget.setEnabled(True)
        self.live_min = self.live_integral_min.value()
        self.reset_integral_data()
        self.timer.timeout.connect(self.update_integral_live)
        self.timer.start(100)

    def live_update_checkbox_changed(self):
        if self.live_update_checkbox.isChecked():
            self.play()
        else:
            self.pause()

    # Now handling actual changes on timer timeout
    def live_integral_min_changed(self):
        self.live_min_changed = True
        # self.live_min = self.live_integral_min.value()
        # There's a smarter way to do this, but for now, reset when min changes
        # self.reset_integral_data()

    def live_integral_step_changed(self):
        self.live_step_changed = True
        # self._temp_auto_spacing = self.live_integral_step.value()
        # self.update_integral_data()

    def live_integral_max_changed(self):
        self.live_max_changed = True
        # if self.live_max_lines_visible < self.live_integral_max.value():
        #    self.live_max_lines_visible = self.live_integral_max.value()
        #    #self.auto_set_spacing() #Need to have it decrease the spacing
        #    self.reset_integral_data()
        # elif self.live_max_lines_visible > self.live_integral_max.value():
        #    self.live_max_lines_visible = self.live_integral_max.value()
        #    self.reset_integral_data()

    def manual_integral_min_changed(self):
        # self.manual_min = self.integral_min.value()
        self.live_min = self.integral_min.value()
        self.reset_integral_data(manual=True)

    def manual_integral_step_changed(self):
        # self.manual_spacing = self.integral_step.value()
        self._temp_auto_spacing = self.integral_step.value()
        self.reset_integral_data(manual=True)

    def manual_integral_max_changed(self):
        # self.manual_max = self.integral_max.value()
        if self.manual_max < self.integral_max.value():
            self.manual_max = self.integral_max.value()
            self.reset_integral_data(manual=True)
        elif self.manual_max > self.integral_max.value():
            self.manual_max = self.integral_max.value()
            self.reset_integral_data(manual=True)

    def tth_line_checkbox_changed(self):
        if self.tth_line_checkbox.isChecked():
            self.tth_line.setPen(255, 255, 255, 150)
        else:
            self.tth_line.setPen(0, 0, 0, 0)


class TabbedArea(QtWidgets.QTabWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings

        self.stats_page = QtWidgets.QWidget()
        self.contour_page = QtWidgets.QWidget()
        self.csim_page = QtWidgets.QWidget()

        self.stats_widget = StatsView(self.stats_page, self.settings)
        self.contour_widget = ContourView(self.contour_page, self.settings)
        self.csim_widget = CSimView(self.csim_page, self.settings)

        self.stats_layout = QtWidgets.QGridLayout()
        self.contour_layout = QtWidgets.QGridLayout()
        self.csim_layout = QtWidgets.QGridLayout()

        # self.skew_checkbox = QtWidgets.QCheckBox("Skew")
        # self.kurtosis_checkbox = QtWidgets.QCheckBox("Kurtosis")
        # self.skew_masked_checkbox = QtWidgets.QCheckBox("Skew Masked")
        # self.kurtosis_masked_checkbox = QtWidgets.QCheckBox("Kurtosis Masked")
        # self.skew_nonzero_checkbox = QtWidgets.QCheckBox("Skew Nonzero")
        # self.kurtosis_nonzero_checkbox = QtWidgets.QCheckBox("Kurtosis Nonzero")
        # self.stats_line_checkbox = QtWidgets.QCheckBox("2Th Line")

        # self.skew_checkbox.setChecked(True)
        # self.kurtosis_checkbox.setChecked(True)
        # self.skew_masked_checkbox.setChecked(True)
        # self.kurtosis_masked_checkbox.setChecked(True)
        # self.skew_nonzero_checkbox.setChecked(True)
        # self.kurtosis_nonzero_checkbox.setChecked(True)

        # self.skew_checkbox.stateChanged.connect(self.skewCheckboxChanged)
        # self.kurtosis_checkbox.stateChanged.connect(self.kurtosisCheckboxChanged)
        # self.skew_masked_checkbox.stateChanged.connect(self.skewMaskedCheckboxChanged)
        # self.kurtosis_masked_checkbox.stateChanged.connect(self.kurtosisMaskedCheckboxChanged)
        # self.skew_nonzero_checkbox.stateChanged.connect(self.skewNonzeroCheckboxChanged)
        # self.kurtosis_nonzero_checkbox.stateChanged.connect(self.kurtosisNonzeroCheckboxChanged)

        # self.stats_line_checkbox.stateChanged.connect(self.statsLineCheckboxChanged)

        self.stats_layout.addWidget(self.stats_widget, 0, 0, 2, 6)
        self.stats_layout.addWidget(self.stats_widget.histogram_type_select, 2, 0)
        # self.stats_layout.addWidget(self.kurtosis_checkbox,2,1)
        # self.stats_layout.addWidget(self.skew_masked_checkbox,2,2)
        # self.stats_layout.addWidget(self.kurtosis_masked_checkbox,2,3)
        # self.stats_layout.addWidget(self.skew_nonzero_checkbox,2,4)
        # self.stats_layout.addWidget(self.kurtosis_nonzero_checkbox,2,5)

        # self.stats_layout.addWidget(self.stats_line_checkbox,3,0)
        self.stats_page.setLayout(self.stats_layout)

        # self.contour_layout.addWidget(self.contour_widget,0,0)
        self.contour_layout.addWidget(self.contour_widget, 0, 0, 5, 6)
        self.contour_layout.addWidget(self.contour_widget.live_update_checkbox, 5, 0)
        self.contour_layout.addWidget(self.contour_widget.tth_line_checkbox, 5, 1)
        self.contour_layout.addWidget(self.contour_widget.integral_select, 5, 2)
        self.contour_layout.addWidget(self.contour_widget.live_integral_min_label, 6, 0)
        self.contour_layout.addWidget(self.contour_widget.live_integral_min, 6, 1)
        self.contour_layout.addWidget(self.contour_widget.live_integral_max_label, 6, 2)
        self.contour_layout.addWidget(self.contour_widget.live_integral_max, 6, 3)
        self.contour_layout.addWidget(
            self.contour_widget.live_integral_step_label, 6, 4
        )
        self.contour_layout.addWidget(self.contour_widget.live_integral_step, 6, 5)
        self.contour_layout.addWidget(self.contour_widget.integral_min_label, 7, 0)
        self.contour_layout.addWidget(self.contour_widget.integral_min, 7, 1)
        self.contour_layout.addWidget(self.contour_widget.integral_max_label, 7, 2)
        self.contour_layout.addWidget(self.contour_widget.integral_max, 7, 3)
        self.contour_layout.addWidget(self.contour_widget.integral_step_label, 7, 4)
        self.contour_layout.addWidget(self.contour_widget.integral_step, 7, 5)
        self.contour_page.setLayout(self.contour_layout)

        self.csim_layout.addWidget(self.csim_widget)
        self.csim_page.setLayout(self.csim_layout)

        self.addTab(self.contour_page, "Contour")
        self.addTab(self.stats_page, "Stats")
        self.addTab(self.csim_page, "Similarity")

    def update_dir(self):
        self.contour_widget.update_dir()
        self.stats_widget.update_dir()
        self.csim_widget.update_dir()

    """
    def skewCheckboxChanged(self,evt):
        if self.skew_checkbox.isChecked():
            self.stats_widget.skew.setPen((255,255,255))
        else:
            self.stats_widget.skew.setPen((0,0,0,0))
    def kurtosisCheckboxChanged(self,evt):
        if self.kurtosis_checkbox.isChecked():
            self.stats_widget.kurtosis.setPen((200,100,0))
        else:
            self.stats_widget.kurtosis.setPen((0,0,0,0))
    def skewMaskedCheckboxChanged(self,evt):
        if self.skew_masked_checkbox.isChecked():
            self.stats_widget.skew_masked.setPen((255,255,255))
        else:
            self.stats_widget.skew_masked.setPen((0,0,0,0))
    def kurtosisMaskedCheckboxChanged(self,evt):
        if self.kurtosis_masked_checkbox.isChecked():
            self.stats_widget.kurtosis_masked.setPen((200,100,0))
        else:
            self.stats_widget.kurtosis_masked.setPen((0,0,0,0))
    def skewNonzeroCheckboxChanged(self,evt):
        if self.skew_nonzero_checkbox.isChecked():
            self.stats_widget.skew_nonzero.setPen((255,255,255))
        else:
            self.stats_widget.skew_nonzero.setPen((0,0,0,0))
    def kurtosisNonzeroCheckboxChanged(self,evt):
        if self.kurtosis_nonzero_checkbox.isChecked():
            self.stats_widget.kurtosis_nonzero.setPen((200,100,0))
        else:
            self.stats_widget.kurtosis_nonzero.setPen((0,0,0,0))
    def statsLineCheckboxChanged(self,evt):
        if self.stats_line_checkbox.isChecked():
            self.stats_widget.stats_line.setPen((0,0,0,150))
        else:
            self.stats_widget.stats_line.setPen((0,0,0,0))
    """


def main_GUI():
    app = QtWidgets.QApplication([])
    larger = KeyPressWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main_GUI()
