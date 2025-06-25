import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import os
import glob
import numpy as np
import pandas as pd

from mainUI.UI_settings import Settings
from corrections_and_maps import q_to_tth


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
        self.spots_count_data = None
        self.spots_area_data = None
        self.spots_intensitymax_data = None
        self.spots_intensitymean_data = None
        self.spots_intensitysum_data = None
        self.scatter_q_bins = None
        self.scatter_tth_bins = None
        self.x_bins = None
        self.y_bins = None
        self.x_axis_type = "tth"

        self.spots_histogram_area_curve = pg.PlotCurveItem(
            fillLevel=0, brush=(0, 0, 255, 80)
        )
        self.spots_histogram_Q_curve = pg.PlotCurveItem(
            fillLevel=0, brush=(0, 255, 0, 80)
        )
        self.spots_histogram_area_Q = pg.ImageItem()
        self.spots_scatter_area = pg.ScatterPlotItem()
        self.spots_scatter_intensitymax = pg.ScatterPlotItem()
        self.spots_scatter_intensitymean = pg.ScatterPlotItem()
        self.spots_scatter_intensitysum = pg.ScatterPlotItem()
        self.spots_area_contour = pg.ImageItem()
        self.spots_Q_contour = pg.ImageItem()
        self.spots_count = self.stats_view.plot()
        self.spots_count.setPen("r")

        self.legend = self.stats_view.addLegend(offset=(-1, 1))

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.stats_view.addItem(self.vLine, ignoreBounds=True)

        # UI
        self.histogram_type_select = QtWidgets.QComboBox()
        self.histogram_type_dict = {
            "Spot Area": self.spots_histogram_area_curve,
            "Spot Q Position": self.spots_histogram_Q_curve,
            "Area vs Q": self.spots_histogram_area_Q,
            "Spot Area vs Time": self.spots_area_contour,
            "Spot Q Position vs Time": self.spots_Q_contour,
            "Area vs Q Scatter": self.spots_scatter_area,
        }
        self.histogram_types = list(self.histogram_type_dict.keys())
        # print(self.histogram_types)
        self.histogram_type_select.addItems(self.histogram_types)
        self.histogram_type_select.currentIndexChanged.connect(
            self.histogram_type_changed
        )
        self.histogram_type_select.setCurrentIndex(5)

    def histogram_type_changed(self, evt):
        self.stats_view.clear()
        self.legend.clear()
        self.stats_view.addItem(self.histogram_type_dict[self.histogram_types[evt]])
        if evt != 2:
            self.legend.addItem(
                self.histogram_type_dict[self.histogram_types[evt]],
                self.histogram_types[evt],
            )
            self.stats_view.addItem(self.vLine, ignoreBounds = True)
        if evt == 5:
            self.legend.addItem(self.spots_scatter_intensitymax, "Max Intensity")
            self.legend.addItem(self.spots_scatter_intensitymean, "Mean Intensity")
            self.legend.addItem(self.spots_scatter_intensitysum, "Summed Intensity")
            self.legend.addItem(self.spots_count, "Spot Count")
            self.stats_view.addItem(self.spots_scatter_intensitymax)
            self.stats_view.addItem(self.spots_scatter_intensitymean)
            self.stats_view.addItem(self.spots_scatter_intensitysum)
            self.stats_view.addItem(self.spots_count)

    def change_x_axis_type(self, evt):
        # tth = 0, Q = 1
        # define a struct for reading that
        if evt == 0:
            self.x_axis_type = "tth"
            self.update_tth()
        elif evt == 1:
            self.x_axis_type = "Q"
            self.update_q()

    def update_stats_data(self):
        stats_infile = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_spots_stats_df.csv"
        )
        stats_df = pd.read_csv(stats_infile)
        # spot_stat_label, area, medianQ, Qbin, on_arc
        stats_df.drop(index = 0, inplace = True)
        self.spots_histogram_Q = stats_df["Qbin"].value_counts().sort_index()
        self.spots_stats_hist, self.area_bins, self.Q_bins = np.histogram2d(stats_df["area"].values, stats_df["medianQ"].values, bins=[50,self.q_bins])
        self.spots_histogram_area = np.sum(self.spots_stats_hist, axis=1)
        # self.spots_histogram_Q = np.sum(self.spots_stats_hist, axis=0)
        self.spots_histogram_area_curve.setData(
            self.area_bins, self.spots_histogram_area, stepMode="center"
        )
        # self.spots_histogram_Q_curve.setData(
        #     self.Q_bins, self.spots_histogram_Q, stepMode="center"
        # )
        # print(self.q_bins[self.spots_histogram_Q.index])
        # print(self.spots_histogram_Q.values)
        self.spots_count_data = self.spots_histogram_Q.values
        self.spots_area_data = stats_df["area"].values
        self.spots_intensitymax_data = stats_df["intensity_max"].values
        self.spots_intensitymean_data = stats_df["intensity_mean"].values
        self.spots_intensitysum_data = stats_df["intensity_sum"].values
        self.scatter_q_bins = stats_df["medianQ"].values
        self.scatter_tth_bins = q_to_tth(self.scatter_q_bins, self.settings.wavelength)
        if self.x_axis_type == "tth":
            self.spots_count.setData(self.tth_bins[self.spots_histogram_Q.index], self.spots_count_data)
            self.spots_scatter_area.setData(self.scatter_tth_bins, self.spots_area_data)
            self.spots_scatter_intensitymax.setData(self.scatter_tth_bins, self.spots_intensitymax_data)
            self.spots_scatter_intensitymean.setData(self.scatter_tth_bins, self.spots_intensitymean_data)
            self.spots_scatter_intensitysum.setData(self.scatter_tth_bins, self.spots_intensitysum_data)
        elif self.x_axis_type == "Q":
            self.spots_count.setData(self.q_bins[self.spots_histogram_Q.index], self.spots_count_data)
            self.spots_scatter_area.setData(self.scatter_q_bins, self.spots_area_data)
            self.spots_scatter_intensitymax.setData(self.scatter_q_bins, self.spots_intensitymax_data)
            self.spots_scatter_intensitymean.setData(self.scatter_q_bins, self.spots_intensitymean_data)
            self.spots_scatter_intensitysum.setData(self.scatter_q_bins, self.spots_intensitysum_data)
        self.spots_histogram_area_Q.setImage(self.spots_stats_hist)
        

    def update_stats_data_old(self):
        stats_infile = os.path.join(
            self.settings.output_directory,
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

    def update_contour_data(self):
        stats_infiles = glob.glob(
            os.path.join(
                self.settings.output_directory,
                "stats",
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]]
                # ) + "*_spots_hist.npy",
                ) + "*_spots_stats_df.csv",
            key = lambda x: (len(x), x)
        )
        print(stats_infiles)

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
        # print(self.q_bins)
        self.tth_bins = q_to_tth(self.q_bins, self.settings.wavelength)
        self.update_stats_data()
        return
    
    def update_tth(self):
        self.spots_count.setData(self.tth_bins[self.spots_histogram_Q.index], self.spots_count_data)
        self.spots_scatter_area.setData(self.scatter_tth_bins, self.spots_area_data)
        self.spots_scatter_intensitymax.setData(self.scatter_tth_bins, self.spots_intensitymax_data)
        self.spots_scatter_intensitymean.setData(self.scatter_tth_bins, self.spots_intensitymean_data)
        self.spots_scatter_intensitysum.setData(self.scatter_tth_bins, self.spots_intensitysum_data)

    def update_q(self):
        self.spots_count.setData(self.q_bins[self.spots_histogram_Q.index], self.spots_count_data)
        self.spots_scatter_area.setData(self.scatter_q_bins, self.spots_area_data)
        self.spots_scatter_intensitymax.setData(self.scatter_q_bins, self.spots_intensitymax_data)
        self.spots_scatter_intensitymean.setData(self.scatter_q_bins, self.spots_intensitymean_data)
        self.spots_scatter_intensitysum.setData(self.scatter_q_bins, self.spots_intensitysum_data)


