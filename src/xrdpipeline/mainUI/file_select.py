"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the directory and file selection widget for the results UI.
"""

import os
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

from mainUI.UI_settings import Settings
from general.file_selection import FileSelectRowWidget
from general.file_name_definitions import add_output_subdirectory


class FileSelectWindow(QtWidgets.QWidget):
    """
    Widget which pops up to display a directory and file select dialog for the
    input directory, output directory, and image control file.
    Updates the settings and sends a signal when all files are selected and "Okay"
    is pressed.
    """

    file_selected = pg.QtCore.Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

        self.image_directory_widget = FileSelectRowWidget("Image Directory:", default_text=self.settings.image_directory, isdir=True)
        self.output_directory_widget = FileSelectRowWidget("Output Directory:", default_text=self.settings.output_directory, isdir=True)
        self.imctrl_file_widget = FileSelectRowWidget(
            "Image Control File:",
            default_text=self.settings.imagecontrol,
            startdir=self.image_directory_widget.file_name.text(),
            ext="Imctrl and PONI files (*.imctrl *.poni)"
        )

        self.okay_button = QtWidgets.QPushButton("Okay")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

        self.file_select_layout = QtWidgets.QVBoxLayout()
        self.file_select_layout.addWidget(self.image_directory_widget)
        self.file_select_layout.addWidget(self.output_directory_widget)
        self.file_select_layout.addWidget(self.imctrl_file_widget)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.okay_button)
        self.button_layout.addWidget(self.cancel_button)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.file_select_layout)
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)

        self.okay_button.released.connect(self.okay_button_pressed)
        self.cancel_button.released.connect(self.cancel_button_pressed)

    def update_shown_info(self):
        self.image_directory_widget.file_name.setText(self.settings.image_directory)
        self.output_directory_widget.file_name.setText(self.settings.output_directory)
        self.imctrl_file_widget.file_name.setText(self.settings.imagecontrol)

    def apply_changes(self):
        self.settings.image_directory = self.image_directory_widget.file_name.text()
        self.settings.output_directory = self.output_directory_widget.file_name.text()
        self.settings.output_directory = add_output_subdirectory(self.settings.output_directory)
        self.settings.imagecontrol = self.imctrl_file_widget.file_name.text()
        self.file_selected.emit()

    def okay_button_pressed(self):
        self.apply_changes()
        self.close()

    def cancel_button_pressed(self):
        self.close()

