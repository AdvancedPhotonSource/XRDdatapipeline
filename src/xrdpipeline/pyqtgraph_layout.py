"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file runs the results UI.
"""

import sys

import PySide6
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from mainUI.main_window import KeyPressWindow

pg.setConfigOptions(imageAxisOrder="row-major")


def main_GUI():
    app = QtWidgets.QApplication([])
    larger = KeyPressWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main_GUI()
