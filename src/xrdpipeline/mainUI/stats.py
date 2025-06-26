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
        self.spots_count_vb = pg.ViewBox()
        self.stats_view.scene().addItem(self.spots_count_vb)
        self.stats_view.showAxis('right')
        self.stats_view.getAxis('right').linkToView(self.spots_count_vb)
        self.spots_count_vb.setXLink(self.stats_view)
        self.stats_view.getAxis('right').setLabel('Spot Count')

        self.spots_count_data = None
        self.spots_area_data = None
        self.spots_intensitymax_data = None
        self.spots_intensitymean_data = None
        self.spots_intensitysum_data = None
        self.scatter_q_bins = None
        self.scatter_tth_bins = None
        self.x_axis_type = "tth"

        self.spots_scatter_area = pg.ScatterPlotItem()
        self.spots_scatter_intensitymax = pg.ScatterPlotItem()
        self.spots_scatter_intensitymean = pg.ScatterPlotItem()
        self.spots_scatter_intensitysum = pg.ScatterPlotItem()
        self.spots_count = pg.PlotDataItem()
        self.spots_count.setPen("r")

        self.legend = self.stats_view.addLegend(offset=(-1, 1))

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.stats_view.addItem(self.vLine, ignoreBounds=True)

        # UI
        self.histogram_type_select = QtWidgets.QComboBox()
        self.histogram_type_dict = {
            "Spot Area": self.spots_scatter_area,
            "Max Intensity": self.spots_scatter_intensitymax,
            "Mean Intensity": self.spots_scatter_intensitymean,
            "Summed Intensity": self.spots_scatter_intensitysum,
        }
        self.histogram_types = list(self.histogram_type_dict.keys())
        # print(self.histogram_types)
        self.histogram_type_select.addItems(self.histogram_types)
        self.histogram_type_select.currentIndexChanged.connect(
            self.histogram_type_changed
        )
        self.histogram_type_select.setCurrentIndex(0)
        self.updateViews()
        self.stats_view.vb.sigResized.connect(self.updateViews)

    def updateViews(self):
        self.spots_count_vb.setGeometry(self.stats_view.vb.sceneBoundingRect())
        self.spots_count_vb.linkedViewChanged(self.stats_view.vb, self.spots_count_vb.XAxis)

    def histogram_type_changed(self, evt):
        self.stats_view.clear()
        self.legend.clear()
        self.stats_view.addItem(self.histogram_type_dict[self.histogram_types[evt]])
        self.legend.addItem(
            self.histogram_type_dict[self.histogram_types[evt]],
            self.histogram_types[evt],
        )
        if evt == 0:
            self.stats_view.getAxis('left').setLabel('Area')
        else:
            self.stats_view.getAxis('left').setLabel('Intensity')
        self.stats_view.addItem(self.vLine, ignoreBounds = True)
        self.legend.addItem(self.spots_count, "Spot Count")
        self.spots_count_vb.addItem(self.spots_count)

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
        self.spots_count_data = np.zeros_like(self.tth_bins)
        self.spots_count_data[self.spots_histogram_Q.index] = self.spots_histogram_Q.values
        self.spots_area_data = stats_df["area"].values
        self.spots_intensitymax_data = stats_df["intensity_max"].values
        self.spots_intensitymean_data = stats_df["intensity_mean"].values
        self.spots_intensitysum_data = stats_df["intensity_sum"].values
        self.scatter_q_bins = stats_df["medianQ"].values
        self.scatter_tth_bins = q_to_tth(self.scatter_q_bins, self.settings.wavelength)
        if self.x_axis_type == "tth":
            self.spots_count.setData(self.tth_bins, self.spots_count_data)
            self.spots_scatter_area.setData(self.scatter_tth_bins, self.spots_area_data)
            self.spots_scatter_intensitymax.setData(self.scatter_tth_bins, self.spots_intensitymax_data)
            self.spots_scatter_intensitymean.setData(self.scatter_tth_bins, self.spots_intensitymean_data)
            self.spots_scatter_intensitysum.setData(self.scatter_tth_bins, self.spots_intensitysum_data)
        elif self.x_axis_type == "Q":
            self.spots_count.setData(self.q_bins, self.spots_count_data)
            self.spots_scatter_area.setData(self.scatter_q_bins, self.spots_area_data)
            self.spots_scatter_intensitymax.setData(self.scatter_q_bins, self.spots_intensitymax_data)
            self.spots_scatter_intensitymean.setData(self.scatter_q_bins, self.spots_intensitymean_data)
            self.spots_scatter_intensitysum.setData(self.scatter_q_bins, self.spots_intensitysum_data)
        # self.spots_histogram_area_Q.setImage(self.spots_stats_hist)


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
        self.histogram_type_changed(self.histogram_type_select.currentIndex())
        return
    
    def update_tth(self):
        self.spots_count.setData(self.tth_bins, self.spots_count_data)
        self.spots_scatter_area.setData(self.scatter_tth_bins, self.spots_area_data)
        self.spots_scatter_intensitymax.setData(self.scatter_tth_bins, self.spots_intensitymax_data)
        self.spots_scatter_intensitymean.setData(self.scatter_tth_bins, self.spots_intensitymean_data)
        self.spots_scatter_intensitysum.setData(self.scatter_tth_bins, self.spots_intensitysum_data)

    def update_q(self):
        self.spots_count.setData(self.q_bins, self.spots_count_data)
        self.spots_scatter_area.setData(self.scatter_q_bins, self.spots_area_data)
        self.spots_scatter_intensitymax.setData(self.scatter_q_bins, self.spots_intensitymax_data)
        self.spots_scatter_intensitymean.setData(self.scatter_q_bins, self.spots_intensitymean_data)
        self.spots_scatter_intensitysum.setData(self.scatter_q_bins, self.spots_intensitysum_data)


