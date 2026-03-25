"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the azim vs Q statistics widget for the results UI.
"""


import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import os
import glob
import numpy as np
import pandas as pd

from mainUI.UI_settings import Settings
from general.corrections_and_maps import q_to_tth

class AzimQView(pg.GraphicsLayoutWidget):
    """
    Widget holding the canvas which shows various statistical information
    about azimuthal and Q span of clusters.
    """
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self.setBackground("w")
        self.azimq_view = self.addPlot(title="")

        self.cluster_azimq_data = None
        self.cluster_area_data = None
        self.cluster_diffazim_data = None
        self.cluster_diffq_data = None
        self.cluster_classifier_data = None
        self.cluster_medianq_data = None
        self.cluster_mediantth_data = None
        self.x_axis_type = "tth"

        self.cluster_scatter_azimq = pg.ScatterPlotItem()
        self.cluster_scatter_area = pg.ScatterPlotItem()
        self.cluster_scatter_diffazim = pg.ScatterPlotItem()
        self.cluster_scatter_diffq = pg.ScatterPlotItem()

        self.spotBrush = pg.mkBrush(self.settings.colors["azim_q_spot"].color)
        self.spotPen = pg.mkPen(self.settings.colors["azim_q_spot"].color)
        self.arcBrush = pg.mkBrush(self.settings.colors["azim_q_arc"].color)
        self.arcPen = pg.mkPen(self.settings.colors["azim_q_arc"].color)
        self.scatter_spot_legend_dummy = pg.ScatterPlotItem()
        self.scatter_spot_legend_dummy.setBrush(self.spotBrush)
        self.scatter_spot_legend_dummy.setPen(self.spotPen)
        self.scatter_arc_legend_dummy = pg.ScatterPlotItem()
        self.scatter_arc_legend_dummy.setBrush(self.arcBrush)
        self.scatter_arc_legend_dummy.setPen(self.arcPen)

        self.legend = self.azimq_view.addLegend(offset=(-1, 1))
        self.legend.addItem(self.scatter_arc_legend_dummy, "Texture Arcs")
        self.legend.addItem(self.scatter_spot_legend_dummy, "Spots")

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.azimq_view.addItem(self.vLine, ignoreBounds=True)

        # UI
        self.histogram_type_select = QtWidgets.QComboBox()
        self.histogram_type_dict = {
            "Azimuthal span / Q span": self.cluster_scatter_azimq,
            "Area": self.cluster_scatter_area,
            "Azimuthal span": self.cluster_scatter_diffazim,
            "Q span": self.cluster_scatter_diffq,
        }
        # QComboBox does not accept dict.keys(), so make a list
        self.histogram_types = list(self.histogram_type_dict.keys())
        self.histogram_type_select.addItems(self.histogram_types)
        self.histogram_type_select.currentIndexChanged.connect(
            self.histogram_type_changed
        )
        self.histogram_type_select.setCurrentIndex(0)

        self.filter_area_label = QtWidgets.QLabel("Minimum cluster area filter (pixels):")
        self.filter_area_spinbox = QtWidgets.QSpinBox()
        self.filter_area_spinbox.setMinimum(0)
        self.filter_area_spinbox.setMaximum(1000)
        self.filter_area_spinbox.valueChanged.connect(self.adjust_area_filter)

    def histogram_type_changed(self, evt):
        """
        Slots to the signal from the histogram type QComboBox.
        When a new histogram type is selected on that dropdown, it passes the index
        as an event. The canvas and legend are cleared, and the new histograms are
        added in.
        Also called when the directory is updated to ensure everything is cleared and
        displayed properly.

        :param evt: Index of the requested histogram type
        """
        self.azimq_view.clear()
        self.azimq_view.addItem(self.histogram_type_dict[self.histogram_types[evt]])
        if evt == 0:
            self.azimq_view.getAxis('left').setLabel('Azimuthal span / Q span (\u00b0\u22c5\u212b)')
        elif evt == 1:
            self.azimq_view.getAxis('left').setLabel('Area (pix)')
        elif evt == 2:
            self.azimq_view.getAxis('left').setLabel('Azimuthal span (\u00b0)')
        elif evt == 3:
            self.azimq_view.getAxis('left').setLabel('Q span (\u212b\u207b\u00b9)')
        self.azimq_view.addItem(self.vLine, ignoreBounds = True)

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
        """
        Read in and update the azim vs Q stats information for this image.
        """
        stats_infile = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_azim_vs_Qs.csv"
        )
        if os.path.exists(stats_infile):
            azim_vs_qs_df = pd.read_csv(stats_infile)
            # label, azim_vs_Q, diff_azim, diff_Q, classifier, medianQ, medianAzim, medianAzim_flipped, area
            self.cluster_azimq_data = azim_vs_qs_df["azim_vs_Q"].values
            self.cluster_area_data = azim_vs_qs_df["area"].values
            self.cluster_diffazim_data = azim_vs_qs_df["diff_azim"].values
            self.cluster_diffq_data = azim_vs_qs_df["diff_Q"].values
            self.cluster_classifier_data = azim_vs_qs_df["classifier"].values
            self.cluster_medianq_data = azim_vs_qs_df["medianQ"].values
            self.cluster_mediantth_data = q_to_tth(self.cluster_medianq_data, self.settings.wavelength)
            self.filter = self.cluster_area_data > self.filter_area_spinbox.value()
            if self.x_axis_type == "tth":
                self.cluster_scatter_azimq.setData(self.cluster_mediantth_data[self.filter], self.cluster_azimq_data[self.filter])
                self.cluster_scatter_area.setData(self.cluster_mediantth_data[self.filter], self.cluster_area_data[self.filter])
                self.cluster_scatter_diffazim.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffazim_data[self.filter])
                self.cluster_scatter_diffq.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffq_data[self.filter])
            elif self.x_axis_type == "Q":
                self.cluster_scatter_azimq.setData(self.cluster_medianq_data[self.filter], self.cluster_azimq_data[self.filter])
                self.cluster_scatter_area.setData(self.cluster_medianq_data[self.filter], self.cluster_area_data[self.filter])
                self.cluster_scatter_diffazim.setData(self.cluster_medianq_data[self.filter], self.cluster_diffazim_data[self.filter])
                self.cluster_scatter_diffq.setData(self.cluster_medianq_data[self.filter], self.cluster_diffq_data[self.filter])
            self.update_colors()
        else:
            # If the file for this image does not exist, clear the canvas.
            self.clear_canvas()

    def clear_canvas(self):
        self.cluster_scatter_azimq.clear()
        self.cluster_scatter_area.clear()
        self.cluster_scatter_diffazim.clear()
        self.cluster_scatter_diffq.clear()

    def update_dir(self):
        self.update_stats_data()
        self.histogram_type_changed(self.histogram_type_select.currentIndex())

    def update_tth(self):
        self.cluster_scatter_azimq.setData(self.cluster_mediantth_data[self.filter], self.cluster_azimq_data[self.filter])
        self.cluster_scatter_area.setData(self.cluster_mediantth_data[self.filter], self.cluster_area_data[self.filter])
        self.cluster_scatter_diffazim.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffazim_data[self.filter])
        self.cluster_scatter_diffq.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffq_data[self.filter])
        for cluster_plot_item in [self.cluster_scatter_azimq, self.cluster_scatter_area, self.cluster_scatter_diffazim, self.cluster_scatter_diffq]:
            cluster_plot_item.setBrush(self.brush)
            cluster_plot_item.setPen(self.pen)

    def update_q(self):
        self.cluster_scatter_azimq.setData(self.cluster_medianq_data[self.filter], self.cluster_azimq_data[self.filter])
        self.cluster_scatter_area.setData(self.cluster_medianq_data[self.filter], self.cluster_area_data[self.filter])
        self.cluster_scatter_diffazim.setData(self.cluster_medianq_data[self.filter], self.cluster_diffazim_data[self.filter])
        self.cluster_scatter_diffq.setData(self.cluster_medianq_data[self.filter], self.cluster_diffq_data[self.filter])
        for cluster_plot_item in [self.cluster_scatter_azimq, self.cluster_scatter_area, self.cluster_scatter_diffazim, self.cluster_scatter_diffq]:
            cluster_plot_item.setBrush(self.brush)
            cluster_plot_item.setPen(self.pen)

    def classifier_brush(self, classifier):
        if classifier == 1:
            return self.spotBrush
        elif classifier == 2:
            return self.arcBrush

    def classifier_pen(self, classifier):
        if classifier == 1:
            return self.spotPen
        elif classifier == 2:
            return self.arcPen

    def adjust_area_filter(self, value):
        self.filter = self.cluster_area_data > value
        if self.x_axis_type == "tth":
            self.cluster_scatter_azimq.setData(self.cluster_mediantth_data[self.filter], self.cluster_azimq_data[self.filter])
            self.cluster_scatter_area.setData(self.cluster_mediantth_data[self.filter], self.cluster_area_data[self.filter])
            self.cluster_scatter_diffazim.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffazim_data[self.filter])
            self.cluster_scatter_diffq.setData(self.cluster_mediantth_data[self.filter], self.cluster_diffq_data[self.filter])
        elif self.x_axis_type == "Q":
            self.cluster_scatter_azimq.setData(self.cluster_medianq_data[self.filter], self.cluster_azimq_data[self.filter])
            self.cluster_scatter_area.setData(self.cluster_medianq_data[self.filter], self.cluster_area_data[self.filter])
            self.cluster_scatter_diffazim.setData(self.cluster_medianq_data[self.filter], self.cluster_diffazim_data[self.filter])
            self.cluster_scatter_diffq.setData(self.cluster_medianq_data[self.filter], self.cluster_diffq_data[self.filter])
        self.update_colors()

    def update_colors(self):
        self.spotBrush = pg.mkBrush(self.settings.colors["azim_q_spot"].color)
        self.spotPen = pg.mkPen(self.settings.colors["azim_q_spot"].color)
        self.arcBrush = pg.mkBrush(self.settings.colors["azim_q_arc"].color)
        self.arcPen = pg.mkPen(self.settings.colors["azim_q_arc"].color)
        self.scatter_spot_legend_dummy.setBrush(self.spotBrush)
        self.scatter_spot_legend_dummy.setPen(self.spotPen)
        self.scatter_arc_legend_dummy.setBrush(self.arcBrush)
        self.scatter_arc_legend_dummy.setPen(self.arcPen)

        self.brush = [self.classifier_brush(x) for x in self.cluster_classifier_data[self.filter]]
        self.pen = [self.classifier_pen(x) for x in self.cluster_classifier_data[self.filter]]
        for cluster_plot_item in [self.cluster_scatter_azimq, self.cluster_scatter_area, self.cluster_scatter_diffazim, self.cluster_scatter_diffq]:
            cluster_plot_item.setBrush(self.brush)
            cluster_plot_item.setPen(self.pen)
