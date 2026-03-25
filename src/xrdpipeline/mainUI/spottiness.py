"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the spottiness widget for the results UI.
"""


import pyqtgraph as pg
import os
import numpy as np
import pandas as pd

from mainUI.UI_settings import Settings
from general.corrections_and_maps import q_to_tth

class SpottinessView(pg.GraphicsLayoutWidget):
    """
    Widget showing statistical information of the second
    azimuthal derivative of the image, binned in two-theta.
    """
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.setBackground("w")
        self.setMinimumHeight(150)
        self.settings = settings
        self.view = self.addPlot(title="")
        self.methods = {
            "Grad median": 0,
            "Grad MAD": 1,
            "Grad mean": 2,
            "Grad STD": 3,
            "Grad MAD-STD": 4,
            "Grad STD/MAD": 5,
        }
        self.line = {}
        self.line_data = {}
        self.tth_bins = []
        self.q_bins = []
        self.axis_type = "tth"
        self.legend = self.view.addLegend(offset=(-1,1))
        for k,v in self.methods.items():
            self.line[k] = self.view.plot()
            self.line_data[k] = None
            self.legend.addItem(self.line[k], k)
        self.update_colors()

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.view.addItem(self.vLine, ignoreBounds=True)
    
    def update_dir(self):
        qbins_filename = os.path.join(
            self.settings.output_directory,
            "stats",
            "qbinedges.npy"
        )
        self.q_bins = []
        if os.path.exists(qbins_filename):
            with open(qbins_filename, 'rb') as infile:
                self.q_bins = np.load(infile)
        else:
            print("Missing q bins file.")
        if len(self.q_bins) > 0:
            self.tth_bins = q_to_tth(self.q_bins, self.settings.wavelength)
        self.update_data()
    
    def update_data(self):
        filename_grad = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_spots_stats_grad.csv",
        )
        self.update_colors()
        if os.path.exists(filename_grad):
            grad_stats = pd.read_csv(filename_grad)
            grad_stats.drop(grad_stats.loc[grad_stats["Qbin"] < 0].index, inplace=True)
            grad_stats.drop(grad_stats.loc[grad_stats["Qbin"] >= len(self.q_bins)].index, inplace=True)
            self.line_data["Grad median"] = grad_stats["median"]
            self.line_data["Grad MAD"] = grad_stats["mad"]
            self.line_data["Grad mean"] = grad_stats["mean"]
            self.line_data["Grad STD"] = grad_stats["std"]
            self.line_data["Grad MAD-STD"] = grad_stats["mad"] - grad_stats["std"]
            self.line_data["Grad STD/MAD"] = grad_stats["std"] / grad_stats["mad"]

            if self.axis_type == "tth":
                self.update_tth()
            elif self.axis_type == "q":
                self.update_q()
            else:
                print("Spottiness: Unknown axis type. Defaulting to 2theta.")
                self.update_tth()
        else:
            for k, v in self.line.items():
                v.clear()
            for k, v in self.line_data.items():
                v = None

    def update_tth(self):
        if len(self.tth_bins) > 0 and self.line_data["Grad median"] is not None:
            self.line["Grad median"].setData(self.tth_bins, self.line_data["Grad median"].values)
            self.line["Grad MAD"].setData(self.tth_bins, self.line_data["Grad MAD"].values)
            self.line["Grad mean"].setData(self.tth_bins, self.line_data["Grad mean"].values)
            self.line["Grad STD"].setData(self.tth_bins, self.line_data["Grad STD"].values)
            self.line["Grad MAD-STD"].setData(self.tth_bins, self.line_data["Grad MAD-STD"].values)
            self.line["Grad STD/MAD"].setData(self.tth_bins, self.line_data["Grad STD/MAD"].values)

    def update_q(self):
        if len(self.q_bins) > 0 and self.line_data["Grad median"] is not None:
            self.line["Grad median"].setData(self.q_bins, self.line_data["Grad median"].values)
            self.line["Grad MAD"].setData(self.q_bins, self.line_data["Grad MAD"].values)
            self.line["Grad mean"].setData(self.q_bins, self.line_data["Grad mean"].values)
            self.line["Grad STD"].setData(self.q_bins, self.line_data["Grad STD"].values)
            self.line["Grad MAD-STD"].setData(self.q_bins, self.line_data["Grad MAD-STD"].values)
            self.line["Grad STD/MAD"].setData(self.q_bins, self.line_data["Grad STD/MAD"].values)

    def change_x_axis_type(self, axis_type):
        if axis_type == 0:
            self.axis_type = "tth"
            self.update_tth()
        elif axis_type == 1:
            self.axis_type = "q"
            self.update_q()
        else:
            print("Spottiness: Unknown axis type. Defaulting to 2theta.")
            self.update_tth()

    def update_colors(self):
        self.line["Grad median"].setPen(self.settings.colors["grad_spottiness_median"].color)
        self.line["Grad MAD"].setPen(self.settings.colors["grad_spottiness_MAD"].color)
        self.line["Grad mean"].setPen(self.settings.colors["grad_spottiness_mean"].color)
        self.line["Grad STD"].setPen(self.settings.colors["grad_spottiness_STD"].color)
        self.line["Grad MAD-STD"].setPen(self.settings.colors["grad_spottiness_diff"].color)
        self.line["Grad STD/MAD"].setPen(self.settings.colors["grad_spottiness_div"].color)

