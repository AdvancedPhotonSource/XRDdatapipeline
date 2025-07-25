import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import os

from mainUI.UI_settings import Settings
from corrections_and_maps import tth_to_q

from dataclasses import dataclass
import warnings

@dataclass
class Integral:
    line: pg.PlotDataItem
    data: np.ndarray
    offset_mult: int


class IntegralView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__()
        # self.setMaximumSize(800,300) # for demo
        self.setMinimumHeight(250)
        # self.directory = directory
        self.settings = settings
        self.setBackground("w")

        self.integral_offset_text = QtWidgets.QLabel()
        self.integral_offset_text.setText("Offset:")
        self.integral_offset = QtWidgets.QSpinBox()
        self.integral_offset.setMinimum(0)
        self.integral_offset.setMaximum(1000000)
        self.integral_offset.setSingleStep(100)
        self.integral_offset.setValue(0)
        self.integral_offset.valueChanged.connect(self.set_integral_plot_data)

        self.integral_view = self.addPlot(title="")
        self.axis_type = 0

        # integral_infile_piece = self.settings.directory + "\\integrals\\" + self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]
        # self.integral_data = np.loadtxt(integral_infile_piece+"_base.xye",skiprows=3)
        # self.masked_integral_data = np.loadtxt(integral_infile_piece+"_closed.xye",skiprows=3)
        # self.spotmasked_integral_data = np.loadtxt(integral_infile_piece+"_closedspotsmasked.xye",skiprows=3)
        # self.texturemasked_integral_data = np.loadtxt(integral_infile_piece+"_closedarcsmasked.xye",skiprows=3)
        self.integral_data = np.empty((self.settings.outChannels, 2))
        self.masked_integral_data = np.empty((self.settings.outChannels, 2))
        self.spotmasked_integral_data = np.empty((self.settings.outChannels, 2))
        self.texturemasked_integral_data = np.empty((self.settings.outChannels, 2))

        # self.base_integral = self.integral_view.plot(self.integral_data[:,0],self.integral_data[:,1],pen="black")
        # self.masked_integral = self.integral_view.plot(self.masked_integral_data[:,0],self.masked_integral_data[:,1] + self.integral_offset.value(),pen="green")
        # self.spotmasked_integral = self.integral_view.plot(self.spotmasked_integral_data[:,0],self.spotmasked_integral_data[:,1] + 2*self.integral_offset.value(),pen="darkcyan")
        # self.texturemasked_integral = self.integral_view.plot(self.texturemasked_integral_data[:,0],self.texturemasked_integral_data[:,1] + 3*self.integral_offset.value(),pen="maroon")
        # self.spots_diff_integral = self.integral_view.plot(self.spotmasked_integral_data[:,0],self.integral_data[:,1]-self.spotmasked_integral_data[:,1],pen=(0,0,0,0)) #draw with an invisible pen to start off
        # self.arcs_diff_integral = self.integral_view.plot(self.texturemasked_integral_data[:,0],self.integral_data[:,1]-self.texturemasked_integral_data[:,1],pen=(0,0,0,0))
        self.base_integral = self.integral_view.plot(
            pen=self.settings.colors["base_line"].color
        )
        self.masked_integral = self.integral_view.plot(
            pen=self.settings.colors["outlier_line"].color
        )
        self.spotmasked_integral = self.integral_view.plot(
            pen=self.settings.colors["spot_line"].color
        )
        self.texturemasked_integral = self.integral_view.plot(
            pen=self.settings.colors["arcs_line"].color
        )
        self.spots_diff_integral = self.integral_view.plot(
            pen=self.settings.colors["minus_spot_line"].color
        )
        self.arcs_diff_integral = self.integral_view.plot(
            pen=self.settings.colors["minus_arcs_line"].color
        )
        self.spots_diff_integral.setVisible(False)
        self.arcs_diff_integral.setVisible(False)

        # Vertical line showing mouse 2theta position from the main image
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.integral_view.addItem(self.vLine, ignoreBounds=True)

        # generalize the integral checkboxes
        # define a pen for each, have checkbox swap between (0,0,0,0) and its pen
        self.base_integral_checkbox = QtWidgets.QCheckBox("Base Integral")
        self.base_integral_checkbox.setChecked(True)
        self.base_integral_checkbox.stateChanged.connect(
            self.base_integral_checkbox_changed
        )

        self.masked_integral_checkbox = QtWidgets.QCheckBox("Outlier Masked")
        self.masked_integral_checkbox.setChecked(True)
        self.masked_integral_checkbox.stateChanged.connect(
            self.masked_integral_checkbox_changed
        )

        self.spotmasked_integral_checkbox = QtWidgets.QCheckBox("Spot Masked")
        self.spotmasked_integral_checkbox.setChecked(True)
        self.spotmasked_integral_checkbox.stateChanged.connect(
            self.spotmasked_integral_checkbox_changed
        )

        self.texturemasked_integral_checkbox = QtWidgets.QCheckBox("Texture Masked")
        self.texturemasked_integral_checkbox.setChecked(True)
        # colorCount, palette, style, repaint
        # self.texturemasked_integral_checkbox.setPalette()
        self.texturemasked_integral_checkbox.stateChanged.connect(
            self.texturemasked_integral_checkbox_changed
        )

        self.spots_diff_integral_checkbox = QtWidgets.QCheckBox(
            "Texture Phases"
        )
        self.spots_diff_integral_checkbox.stateChanged.connect(
            self.spots_diff_integral_checkbox_changed
        )
        self.arcs_diff_integral_checkbox = QtWidgets.QCheckBox(
            "Spot Phases"
        )
        self.arcs_diff_integral_checkbox.stateChanged.connect(
            self.arcs_diff_integral_checkbox_changed
        )

        self.sqrt_checkbox = QtWidgets.QCheckBox("Sqrt(y)")
        self.sqrt_checkbox.stateChanged.connect(self.sqrt_toggle)

        # set an order
        # or set the text color to a faded gray when it's toggled off to show it can be there and still keep the order
        self.legend = self.integral_view.addLegend(offset=(-1, 1))
        self.legend.addItem(self.base_integral, "Base Integral")
        self.legend.addItem(self.masked_integral, "Outlier Masked")
        self.legend.addItem(self.spotmasked_integral, "Spot Masked")
        self.legend.addItem(self.texturemasked_integral, "Texture Masked")
        self.legend.addItem(self.spots_diff_integral, "Texture Phases")
        self.legend.addItem(self.arcs_diff_integral, "Spot Phases")

    def update_dir(self):
        # print("Integrals: updating directory")
        self.integral_data = np.empty((self.settings.outChannels, 2))
        self.masked_integral_data = np.empty((self.settings.outChannels, 2))
        self.spotmasked_integral_data = np.empty((self.settings.outChannels, 2))
        self.texturemasked_integral_data = np.empty((self.settings.outChannels, 2))
        self.base_integral.setPen(self.settings.colors["base_line"].color)
        self.masked_integral.setPen(self.settings.colors["outlier_line"].color)
        self.spotmasked_integral.setPen(self.settings.colors["spot_line"].color)
        self.texturemasked_integral.setPen(self.settings.colors["arcs_line"].color)
        self.arcs_diff_integral.setPen(self.settings.colors["minus_arcs_line"].color)
        self.spots_diff_integral.setPen(self.settings.colors["minus_spot_line"].color)
        self.update_integral_data()

    def update_integral_data(self):
        # print("Integrals: updating data")
        # global tiflist, keylist, curr_key, curr_pos
        integral_infile_piece = os.path.join(
            self.settings.output_directory,
            "integrals",
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]
        )
        integrals_dict = {
            "_base.chi": self.integral_data,
            "_om.chi": self.masked_integral_data,
            "_spotsmasked.chi": self.spotmasked_integral_data,
            "_arcsmasked.chi": self.texturemasked_integral_data,
            # "_closed.xye": self.closed_integral_data,
            # "_closedarcsmasked.xye": self.closedarcs_integral_data,
            # "_closedspotsmasked.xye": self.closedspots_integral_data,
            # "_closed_pytorch.xye": self.pytorch_integral_data,
        }
        for ext, vals in integrals_dict.items():
            # print("Integrals: loading data for {0}".format(ext))
            try:
                new_vals = np.loadtxt(integral_infile_piece + ext, skiprows=4)
            except:
                print(
                    "Exception reading in integral data for {0}. Setting to 0.".format(
                        integral_infile_piece + ext
                    )
                )
                vals[:, :] = 0
            if new_vals.shape != vals.shape:
                print(
                    "Shape of incoming data {0} not equal to config shape {1} for integral {2}. Setting values to 0. If the config shape looks incorrect, go to Settings->outChannels and set it to {3}.".format(
                        new_vals.shape,
                        vals.shape,
                        integral_infile_piece + ext,
                        new_vals.shape[0],
                    )
                )
                vals[:, :] = 0
            try:
                vals[:, :] = new_vals
            except:
                print(
                    "Exception setting integral data for {0}. Setting to 0.".format(
                        integral_infile_piece, ext
                    )
                )

        self.set_integral_plot_data()        

    def sqrt_toggle(self, evt):
        if self.sqrt_checkbox.isChecked():
            # Toggle offset
            # self.integral_offset.valueChanged.disconnect()
            # self.integral_offset.setValue(int(self.integral_offset.value()/10))
            self.integral_offset.setValue(
                int(np.round(np.sqrt(self.integral_offset.value()), decimals=-1))
            )
            # self.integral_offset.valueChanged.connect(self.update_integral_offset)
            self.integral_offset.setSingleStep(10)
            # Toggle data. Diff integrals have some negative values.
            self.set_integral_plot_data()
            # self.spots_diff_integral.setData(self.spotmasked_integral_data[:,0],np.sqrt(self.integral_data[:,1]-self.spotmasked_integral_data[:,1]))
            # self.arcs_diff_integral.setData(self.texturemasked_integral_data[:,0],np.sqrt(self.integral_data[:,1]-self.texturemasked_integral_data[:,1]))

        else:
            # Toggle offset
            # self.integral_offset.setValue(self.integral_offset.value()*10)
            self.integral_offset.setValue(
                int(np.round(self.integral_offset.value() ** 2, decimals=-2))
            )
            self.integral_offset.setSingleStep(100)
            # Toggle data. Diff integrals have some negative values.
            self.set_integral_plot_data()
            # self.spots_diff_integral.setData(self.spotmasked_integral_data[:,0],self.integral_data[:,1]-self.spotmasked_integral_data[:,1])
            # self.arcs_diff_integral.setData(self.texturemasked_integral_data[:,0],self.integral_data[:,1]-self.texturemasked_integral_data[:,1])

    def change_x_axis_type(self, axis_type):
        # Will be passed index of axis type
        # 2 theta = 0, Q = 1
        self.axis_type = axis_type
        # self.wavelength = wavelength
        self.update_integral_data()

    def set_integral_plot_data(self):
        base = Integral(self.base_integral, self.integral_data, 0)
        masked = Integral(self.masked_integral, self.masked_integral_data, 1)
        spot = Integral(self.spotmasked_integral, self.spotmasked_integral_data, 2)
        texture = Integral(self.texturemasked_integral, self.texturemasked_integral_data, 3)

        normal_integrals = [base, masked, spot, texture]
        if self.axis_type == 0:
            if self.sqrt_checkbox.isChecked():
                for integral in normal_integrals:
                    integral.line.setData(
                        integral.data[:,0],
                        np.sqrt(integral.data[:,1]) + integral.offset_mult * self.integral_offset.value()
                    )
                # Difference integrals have some negative values
                # Still outputs a RuntimeWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.spots_diff_integral.setData(
                        self.texturemasked_integral_data[:, 0],
                        np.where(self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1] > 0,
                                np.sqrt(self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1]),
                                -np.sqrt(self.texturemasked_integral_data[:, 1] - self.integral_data[:, 1])),
                    )
                    self.arcs_diff_integral.setData(
                        self.spotmasked_integral_data[:, 0],
                        np.where(self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1] > 0,
                                np.sqrt(self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1]),
                                -np.sqrt(self.spotmasked_integral_data[:, 1] - self.integral_data[:, 1])),
                    )
            else:
                for integral in normal_integrals:
                    integral.line.setData(
                        integral.data[:,0],
                        integral.data[:,1] + integral.offset_mult * self.integral_offset.value()
                    )
                self.spots_diff_integral.setData(
                    self.texturemasked_integral_data[:, 0],
                    self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1],
                )
                self.arcs_diff_integral.setData(
                    self.spotmasked_integral_data[:, 0],
                    self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1],
                )

        elif self.axis_type == 1:
            if self.sqrt_checkbox.isChecked():
                for integral in normal_integrals:
                    integral.line.setData(
                        tth_to_q(integral.data[:,0], self.settings.wavelength),
                        np.sqrt(integral.data[:,1]) + integral.offset_mult * self.integral_offset.value()
                    )
                # Difference integrals have some negative values
                # Still outputs a RuntimeWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.spots_diff_integral.setData(
                        tth_to_q(self.texturemasked_integral_data[:, 0], self.settings.wavelength),
                        np.where(self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1] > 0,
                                np.sqrt(self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1]),
                                -np.sqrt(self.texturemasked_integral_data[:, 1] - self.integral_data[:, 1])),
                    )
                    self.arcs_diff_integral.setData(
                        tth_to_q(
                            self.spotmasked_integral_data[:, 0], self.settings.wavelength
                        ),
                        np.where(self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1] > 0,
                                np.sqrt(self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1]),
                                -np.sqrt(self.spotmasked_integral_data[:, 1] - self.integral_data[:, 1])),
                    )
            else:
                for integral in normal_integrals:
                    integral.line.setData(
                        tth_to_q(integral.data[:,0], self.settings.wavelength),
                        integral.data[:,1] + integral.offset_mult * self.integral_offset.value()
                    )
                self.spots_diff_integral.setData(
                    tth_to_q(self.texturemasked_integral_data[:, 0], self.settings.wavelength),
                    self.integral_data[:, 1] - self.texturemasked_integral_data[:, 1],
                )
                self.arcs_diff_integral.setData(
                    tth_to_q(
                        self.spotmasked_integral_data[:, 0], self.settings.wavelength
                    ),
                    self.integral_data[:, 1] - self.spotmasked_integral_data[:, 1],
                )

    def base_integral_checkbox_changed(self):
        if self.base_integral_checkbox.isChecked():
            # self.base_integral.setPen(("black"))
            self.base_integral.setVisible(True)
        else:
            # self.base_integral.setPen((255,255,255,0))
            self.base_integral.setVisible(False)

    def masked_integral_checkbox_changed(self):
        if self.masked_integral_checkbox.isChecked():
            # self.masked_integral.setPen("green")
            self.masked_integral.setVisible(True)
        else:
            # self.masked_integral.setPen((200,100,0,0))
            self.masked_integral.setVisible(False)

    def spotmasked_integral_checkbox_changed(self):
        if self.spotmasked_integral_checkbox.isChecked():
            # self.spotmasked_integral.setPen("darkcyan")
            self.spotmasked_integral.setVisible(True)
        else:
            # self.spotmasked_integral.setPen((0,255,0,0))
            self.spotmasked_integral.setVisible(False)

    def texturemasked_integral_checkbox_changed(self):
        if self.texturemasked_integral_checkbox.isChecked():
            # self.texturemasked_integral.setPen("maroon")
            self.texturemasked_integral.setVisible(True)
        else:
            # self.texturemasked_integral.setPen((0,0,255,0))
            self.texturemasked_integral.setVisible(False)

    def spots_diff_integral_checkbox_changed(self):
        if self.spots_diff_integral_checkbox.isChecked():
            # self.spots_diff_integral.setPen(self.settings.colors["minus_spot_line"].color)
            self.spots_diff_integral.setVisible(True)
        else:
            # self.spots_diff_integral.setPen(0,0,0,0)
            self.spots_diff_integral.setVisible(False)

    def arcs_diff_integral_checkbox_changed(self):
        if self.arcs_diff_integral_checkbox.isChecked():
            # self.arcs_diff_integral.setPen(self.settings.colors["minus_arcs_line"].color)
            self.arcs_diff_integral.setVisible(True)
        else:
            # self.arcs_diff_integral.setPen(0,0,0,0)
            self.arcs_diff_integral.setVisible(False)

