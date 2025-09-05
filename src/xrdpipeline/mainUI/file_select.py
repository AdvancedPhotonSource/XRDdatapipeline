"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the file selection widget for the results UI.
"""


from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

from mainUI.UI_settings import Settings


class FileSelectWindow(QtWidgets.QWidget):

    file_selected = pg.QtCore.Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

        self.image_directory_label = QtWidgets.QLabel("Image Directory:")
        self.image_directory_text = QtWidgets.QLineEdit(self.settings.image_directory)
        self.image_directory_browse_button = QtWidgets.QPushButton("Browse...")

        self.output_directory_label = QtWidgets.QLabel("Output Directory:")
        self.output_directory_text = QtWidgets.QLineEdit(self.settings.output_directory)
        self.output_directory_browse_button = QtWidgets.QPushButton("Browse...")

        self.imctrl_file_label = QtWidgets.QLabel("Image Control File:")
        self.imctrl_file_text = QtWidgets.QLineEdit(self.settings.imagecontrol)
        self.imctrl_file_browse_button = QtWidgets.QPushButton("Browse...")

        self.okay_button = QtWidgets.QPushButton("Okay")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

        self.file_select_layout = QtWidgets.QGridLayout()
        self.file_select_layout.addWidget(self.image_directory_label, 0, 0)
        self.file_select_layout.addWidget(self.image_directory_text, 0, 1)
        self.file_select_layout.addWidget(self.image_directory_browse_button, 0, 2)
        self.file_select_layout.addWidget(self.output_directory_label, 1, 0)
        self.file_select_layout.addWidget(self.output_directory_text, 1, 1)
        self.file_select_layout.addWidget(self.output_directory_browse_button, 1, 2)
        self.file_select_layout.addWidget(self.imctrl_file_label, 2, 0)
        self.file_select_layout.addWidget(self.imctrl_file_text, 2, 1)
        self.file_select_layout.addWidget(self.imctrl_file_browse_button, 2, 2)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.okay_button)
        self.button_layout.addWidget(self.cancel_button)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.file_select_layout)
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)

        self.image_directory_browse_button.released.connect(self.browse_image_dir)
        self.output_directory_browse_button.released.connect(self.browse_output_dir)
        self.imctrl_file_browse_button.released.connect(self.browse_imctrl)
        self.okay_button.released.connect(self.okay_button_pressed)
        self.cancel_button.released.connect(self.cancel_button_pressed)

    def update_shown_info(self):
        self.image_directory_text.setText(self.settings.image_directory)
        self.output_directory_text.setText(self.settings.output_directory)
        self.imctrl_file_text.setText(self.settings.imagecontrol)

    def browse_image_dir(self):
        image_directory_name = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Image Directory")
        self.image_directory_text.setText(image_directory_name)

    def browse_output_dir(self):
        output_directory_name = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Image Directory")
        self.output_directory_text.setText(output_directory_name)

    def browse_imctrl(self):
        imctrl_file_name = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose Configuration File",
            self.image_directory_text.text(),
            "Imctrl and PONI files (*.imctrl *.poni)"
        )
        self.imctrl_file_text.setText(imctrl_file_name[0])

    def apply_changes(self):
        self.settings.image_directory = self.image_directory_text.text()
        self.settings.output_directory = self.output_directory_text.text()
        self.settings.imagecontrol = self.imctrl_file_text.text()
        self.file_selected.emit()

    def okay_button_pressed(self):
        self.apply_changes()
        self.close()

    def cancel_button_pressed(self):
        self.close()

