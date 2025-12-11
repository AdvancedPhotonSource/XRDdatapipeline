"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file runs the results UI.
"""

import sys
import argparse

import PySide6
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from mainUI.main_window import KeyPressWindow

pg.setConfigOptions(imageAxisOrder="row-major")


def main_GUI(
        image_directory=".",
        output_directory=".",
        imagecontrol="",
        show_directory_prompt=True,
        read_outChannels_from_imctrl=True,
        outChannels=0,
        ):
    app = QtWidgets.QApplication([])
    window = KeyPressWindow(
        image_directory=image_directory,
        output_directory=output_directory,
        imagecontrol=imagecontrol,
        show_directory_prompt=show_directory_prompt,
    )
    if not read_outChannels_from_imctrl:
        window.settings.outChannels = outChannels
        window.update_settings()
        window.update_dir(reread_imctrl_outChannels = read_outChannels_from_imctrl)
    sys.exit(app.exec())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_directory', default=".")
    parser.add_argument('-o', '--output_directory', default=".")
    parser.add_argument('-c', '--imctrl', default="")
    parser.add_argument('-s', '--show_directory_prompt', action='store_false')
    parser.add_argument('-r', '--read_outChannels_from_imctrl', action='store_false')
    parser.add_argument('-b', '--outChannels', type=int, default=0)
    args = parser.parse_args()
    main_GUI(
        image_directory=args.image_directory,
        output_directory=args.output_directory,
        imagecontrol=args.imctrl,
        show_directory_prompt=args.show_directory_prompt,
        read_outChannels_from_imctrl=args.read_outChannels_from_imctrl,
        outChannels=args.outChannels,
    )
