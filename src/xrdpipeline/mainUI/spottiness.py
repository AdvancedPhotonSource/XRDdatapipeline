import pyqtgraph as pg
import os
import numpy as np
import pandas as pd

from mainUI.UI_settings import Settings
from corrections_and_maps import q_to_tth

class SpottinessView(pg.GraphicsLayoutWidget):
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
            self.line_data[k] = []
            self.legend.addItem(self.line[k], k)
        self.line["Grad median"].setPen("hotpink")
        self.line["Grad MAD"].setPen("cyan")
        self.line["Grad mean"].setPen("r")
        self.line["Grad STD"].setPen("g")
        self.line["Grad MAD-STD"].setPen("b")
        self.line["Grad STD/MAD"].setPen("b")

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.view.addItem(self.vLine, ignoreBounds=True)
    
    def update_dir(self):
        qbins_filename = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key][:-1] + "_qbinedges.npy"
        )
        qbins_alt1 = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key][:-1] + "-00000_qbinedges.npy"
        )
        qbins_alt2 = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key][:-6] + "_qbinedges.npy"
        )
        # print(qbins_filename)
        if os.path.exists(qbins_filename):
            with open(qbins_filename, 'rb') as infile:
                self.q_bins = np.load(infile)
        elif os.path.exists(qbins_alt1):
            with open(qbins_alt1, 'rb') as infile:
                self.q_bins = np.load(infile)
        elif os.path.exists(qbins_alt2):
            with open(qbins_alt2, 'rb') as infile:
                self.q_bins = np.load(infile)
        else:
            print("Missing q bins file.")
        self.tth_bins = q_to_tth(self.q_bins, self.settings.wavelength)
        self.update_data()
    
    def update_data(self):
        filename_df = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key] + self.settings.curr_num + "_spots_stats_df.csv"
        )
        filename_grad = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key] + self.settings.curr_num + "_spots_stats_grad.csv"
        )
        df_stats = pd.read_csv(filename_df)
        df_counts = df_stats["Qbin"].value_counts().sort_index()
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

    def update_tth(self):
        self.line["Grad median"].setData(self.tth_bins, self.line_data["Grad median"].values)
        self.line["Grad MAD"].setData(self.tth_bins, self.line_data["Grad MAD"].values)
        self.line["Grad mean"].setData(self.tth_bins, self.line_data["Grad mean"].values)
        self.line["Grad STD"].setData(self.tth_bins, self.line_data["Grad STD"].values)
        self.line["Grad MAD-STD"].setData(self.tth_bins, self.line_data["Grad MAD-STD"].values)
        self.line["Grad STD/MAD"].setData(self.tth_bins, self.line_data["Grad STD/MAD"].values)

    def update_q(self):
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


