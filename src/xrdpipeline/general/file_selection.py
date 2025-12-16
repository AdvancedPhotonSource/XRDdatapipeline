"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the file selection widget used in multiple other files.
"""

from pyqtgraph.Qt import QtWidgets

class FileSelectRowWidget(QtWidgets.QWidget):
    """
    File/directory selection row widget, containing a button which pulls up a
    file selection dialog and a label which fills with the selected result.
    """
    def __init__(self, label, default_text=None, isdir=False, startdir=".", ext=None, free_last_column=False, min_width = 600):
        super().__init__()
        self.setMinimumWidth(min_width)
        self.file_select_button = QtWidgets.QPushButton(label)
        self.file_name = QtWidgets.QLabel(default_text)
        self.isdir = isdir
        self.startdir = startdir
        self.ext = ext

        self.setLayout(QtWidgets.QGridLayout())
        self.layout().addWidget(self.file_select_button, 0, 0)
        if free_last_column:
            self.layout().addWidget(self.file_name, 0, 1, 1, 2)
        else:
            self.layout().addWidget(self.file_name, 0, 1, 1, 3)

        self.file_select_button.released.connect(self.file_select_button_pressed)

    def file_select_button_pressed(self):
        self.select_file()

    def select_file(self):
        if self.isdir:
            location = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Select Directory"
            )
            self.file_name.setText(location)
        else:
            location = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select File", self.startdir, self.ext
            )
            self.file_name.setText(location[0])