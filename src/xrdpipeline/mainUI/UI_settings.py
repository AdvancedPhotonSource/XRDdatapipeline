from dataclasses import dataclass, field
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
import re


@dataclass
class ColorSettings:
    name: str
    color: str
    default_color: str = field(init=False)

    def __post_init__(self):
        self.default_color = self.color


@dataclass
class Settings:
    image_directory: str
    output_directory: str
    imagecontrol: str
    image_size: tuple
    wavelength: float
    outChannels: int
    keylist: list
    tiflist: dict
    curr_num: str
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
                "Base \u2013 Spot Masked Integral Line", "blue"
            ),
            "minus_arcs_line": ColorSettings(
                "Base \u2013 Texture Masked Integral Line", "hotpink"
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

        self.image_directory_label = QtWidgets.QLabel("Image Directory:")
        self.image_directory_text = QtWidgets.QLineEdit(self.settings.image_directory)
        self.image_directory_browse_button = QtWidgets.QPushButton("Browse...")

        self.output_directory_label = QtWidgets.QLabel("Output Directory:")
        self.output_directory_text = QtWidgets.QLineEdit(self.settings.output_directory)
        self.output_directory_browse_button = QtWidgets.QPushButton("Browse...")

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
        self.settings_layout.addWidget(self.image_directory_label, 0, 0)
        self.settings_layout.addWidget(self.image_directory_text, 0, 1)
        self.settings_layout.addWidget(self.image_directory_browse_button, 0, 2)
        self.settings_layout.addWidget(self.output_directory_label, 1, 0)
        self.settings_layout.addWidget(self.output_directory_text, 1, 1)
        self.settings_layout.addWidget(self.output_directory_browse_button, 1, 2)
        self.settings_layout.addWidget(self.imctrl_file_label, 2, 0)
        self.settings_layout.addWidget(self.imctrl_file_text, 2, 1)
        self.settings_layout.addWidget(self.imctrl_file_browse_button, 2, 2)
        self.settings_layout.addWidget(self.wavelength_label, 3, 0)
        self.settings_layout.addWidget(self.wavelength_text, 3, 1, 1, 2)
        self.settings_layout.addWidget(self.outChannels_label, 4, 0)
        self.settings_layout.addWidget(self.outChannels_text, 4, 1, 1, 2)

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

        self.image_directory_browse_button.released.connect(self.browse_image_dir)
        self.output_directory_browse_button.released.connect(self.browse_output_dir)
        self.imctrl_file_browse_button.released.connect(self.browse_imctrl)
        self.apply_button.released.connect(self.apply_button_pressed)
        self.okay_button.released.connect(self.okay_button_pressed)
        self.cancel_button.released.connect(self.cancel_button_pressed)

    def update_shown_info(self):
        self.image_directory_text.setText(self.settings.image_directory)
        self.output_directory_text.setText(self.settings.output_directory)
        self.imctrl_file_text.setText(self.settings.imagecontrol)
        self.wavelength_text.setText(str(self.settings.wavelength))
        self.outChannels_text.setText(str(self.settings.outChannels))

    def browse_image_dir(self):
        image_directory_name = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Image Directory"
        )
        self.image_directory_text.setText(image_directory_name)

    def browse_output_dir(self):
        output_directory_name = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Output Directory"
        )
        self.output_directory_text.setText(output_directory_name)

    def browse_imctrl(self):
        imctrl_file_name = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose Configuration File",
            self.image_directory_text.text(),
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
        self.settings.image_directory = self.image_directory_text.text()
        self.settings.output_directory = self.output_directory_text.text()
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

