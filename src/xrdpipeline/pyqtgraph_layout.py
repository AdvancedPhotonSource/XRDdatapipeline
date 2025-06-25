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
