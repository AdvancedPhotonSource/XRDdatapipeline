import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import os
import glob
import numpy as np
import time
from enum import Enum

from mainUI.UI_settings import Settings
from corrections_and_maps import tth_to_q

Viewtype = Enum('Viewtype', names=[('Contour',0),('Waterfall',1)])
viewtypes = [
    "Contour",
    "Waterfall",
]

class ContourView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.setMinimumHeight(150)
        # self.directory = directory
        self.settings = settings
        self.live_max_lines_visible = 100
        self.live_spacing = 1
        self._temp_auto_spacing = 1
        self.live_min = 0
        self.live_min_changed = False
        self.live_step_changed = False
        self.live_max_changed = False
        self.manual_min = 0
        self.manual_max = 100
        self.manual_spacing = 1
        self.integral_extension = "_om.chi"
        self.integral_data = []
        self.automatically_set_spacing = True
        # Intended to be "Should this zoom out (change step size) when more images appear than the max, or should it start scrolling?" Not yet implemented.
        self.modes = ["zoom", "scroll"]
        self.mode = "zoom"
        # xvals will swap between tthvals and qvals
        self.x_axis_type = 0  # 0 = tth, 1 = Q
        self.tthvals = []
        self.qvals = []
        self.yvals = []
        # self.wavelength = self.settings.wavelength
        self.timer = QtCore.QTimer()

        self.view = self.addPlot(title="")
        self.view.setAspectLocked(False)
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)
        # self.image_data = tf.imread(tiflist[keylist[curr_key]][curr_pos] + ".tif")
        self.contour_image = pg.ImageItem()
        self.contour_waterfall = []
        # CET-R1, CET-R2, CET-R3
        self.waterfall_cmap = pg.colormap.get("CET-R3", skipCache=True)
        self.view.addItem(self.contour_image)
        self.intensityBar = pg.HistogramLUTItem()
        self.intensityBar.setImageItem(self.contour_image)
        self.intensityBar.gradient.setColorMap(self.cmap)
        self.intensityBar.gradient.showTicks(show=False)
        self.addItem(self.intensityBar)

        self.tth_line = pg.InfiniteLine(angle=90, movable=False)
        self.view.addItem(self.tth_line, ignoreBounds=True)
        self.horiz_line = pg.InfiniteLine(angle=0, movable=False)
        self.view.addItem(self.horiz_line, ignoreBounds=True)

        # self.update_integral_list()

        self.live_update_checkbox = QtWidgets.QCheckBox("Live update")
        self.live_update_checkbox.stateChanged.connect(
            self.live_update_checkbox_changed
        )
        self.tth_line_checkbox = QtWidgets.QCheckBox("2Th Line")
        self.tth_line_checkbox.stateChanged.connect(self.tth_line_checkbox_changed)

        self.integral_select = QtWidgets.QComboBox()
        # self.integral_types = ["Base","Outlier Masked","Closed Mask"]
        self.integral_type_dict = {
            "Base": "_base.chi",
            "Outlier Mask": "_om.chi",
            "Spot Mask": "_spotsmasked.chi",
            "Texture Mask": "_arcsmasked.chi",
        }
        self.integral_types = list(self.integral_type_dict.keys())
        self.integral_select.addItems(self.integral_types)
        self.integral_select.setCurrentIndex(1)
        self.integral_select.currentIndexChanged.connect(self.integral_type_changed)

        self.viewtype_select = QtWidgets.QComboBox()
        self.viewtype_select.addItems(viewtypes)
        self.viewtype_select.setCurrentIndex(0)
        self.viewtype_select.currentIndexChanged.connect(self.viewtype_changed)

        self.offset_label = QtWidgets.QLabel("Offset:")
        self.offset = QtWidgets.QSpinBox()
        self.offset.setMaximum(1000000)
        self.offset.setValue(1000)
        self.offset.setSingleStep(1000)
        self.offset.valueChanged.connect(self.offset_changed)

        self.live_controls_list = []
        self.live_integral_min_label = QtWidgets.QLabel()
        self.live_integral_min_label.setText("Min:")
        self.live_controls_list.append(self.live_integral_min_label)
        self.live_integral_min = QtWidgets.QSpinBox()
        self.live_integral_min.setMinimum(0)
        self.live_integral_min.valueChanged.connect(self.live_integral_min_changed)
        self.live_controls_list.append(self.live_integral_min)
        self.live_integral_max_label = QtWidgets.QLabel()
        self.live_integral_max_label.setText("Max integrals drawn:")
        self.live_controls_list.append(self.live_integral_max_label)
        self.live_integral_max = QtWidgets.QSpinBox()
        self.live_integral_max.setMinimum(1)
        self.live_integral_max.setMaximum(10000)
        self.live_integral_max.setValue(100)
        self.live_integral_max.valueChanged.connect(self.live_integral_max_changed)
        self.live_controls_list.append(self.live_integral_max)
        self.live_integral_step_label = QtWidgets.QLabel()
        self.live_integral_step_label.setText("Step size:")
        self.live_controls_list.append(self.live_integral_step_label)
        self.live_integral_step = QtWidgets.QSpinBox()
        self.live_integral_step.setMinimum(1)
        self.live_integral_step.valueChanged.connect(self.live_integral_step_changed)
        self.live_controls_list.append(self.live_integral_step)

        for widget in self.live_controls_list:
            # widget.setEnabled(False)
            widget.hide()

        self.manual_controls_list = []
        self.integral_min_label = QtWidgets.QLabel()
        self.integral_min_label.setText("Min:")
        self.manual_controls_list.append(self.integral_min_label)
        self.integral_min = QtWidgets.QSpinBox()
        self.integral_min.setMinimum(0)
        self.integral_min.valueChanged.connect(self.manual_integral_min_changed)
        self.manual_controls_list.append(self.integral_min)
        self.integral_max_label = QtWidgets.QLabel()
        self.integral_max_label.setText("Max:")
        self.manual_controls_list.append(self.integral_max_label)
        self.integral_max = QtWidgets.QSpinBox()
        self.integral_max.setMinimum(1)
        self.integral_max.setMaximum(10000)
        self.integral_max.setValue(100)
        self.integral_max.valueChanged.connect(self.manual_integral_max_changed)
        self.manual_controls_list.append(self.integral_max)
        self.integral_step_label = QtWidgets.QLabel()
        self.integral_step_label.setText("Step size:")
        self.manual_controls_list.append(self.integral_step_label)
        self.integral_step = QtWidgets.QSpinBox()
        self.integral_step.setMinimum(1)
        self.integral_step.valueChanged.connect(self.manual_integral_step_changed)
        self.manual_controls_list.append(self.integral_step)

    def update_dir(self):
        self.reset_integral_data(reset_z=True)

    def update_integral_list(self, reset_z=False, manual = False):
        # global keylist, curr_key
        self.integral_filelist = sorted(
            glob.glob(
                os.path.join(
                    self.settings.output_directory,
                    "integrals",
                    self.settings.keylist[self.settings.curr_key]
                )
                + "*"
                + self.integral_extension
            ),
            key = lambda x: (len(x), x)
        )
        #Pop the last element of the list if it's been created in the past half second to avoid reading it while it is written
        #Test files showing 0.02 seconds from creation time to modification time
        #print(os.path.getmtime(self.integral_filelist[-1]) - os.path.getctime(self.integral_filelist[-1]))
        if (len(self.integral_filelist) > 0) and (
            time.time() - os.path.getctime(self.integral_filelist[-1]) < 0.5
        ):
            self.integral_filelist.pop()
        if self.automatically_set_spacing:
            self.auto_set_spacing()
        if manual:
            self.update_integral_data_manual()
        else:
            self.update_integral_data(reset_z=reset_z)

    def auto_set_spacing(self):
        # while len(self.integral_filelist) >= (self.max_lines_visible + 1)*self.requested_spacing:
        #    self.requested_spacing *= 2
        self._temp_auto_spacing = self.live_spacing
        while len(self.integral_filelist[self.live_min :: self._temp_auto_spacing]) >= (
            self.live_max_lines_visible + 1
        ):
            self._temp_auto_spacing *= 2

    def append_integral_data(self, max=None):
        # append data from files beyond cur_len*spacing
        cur_len = len(self.integral_data)
        if max == None:
            file_subset = self.integral_filelist[self.live_min :: self.live_spacing]
        else:
            file_subset = self.integral_filelist[
                self.live_min : max : self.live_spacing
            ]
        for i in range(cur_len, len(file_subset)):
            data = np.loadtxt(file_subset[i], skiprows=4)
            if len(self.tthvals) == 0:
                self.tthvals = data[:, 0]
                if self.settings.wavelength != 0:
                    self.qvals = tth_to_q(self.tthvals, self.settings.wavelength)
                # self.surface_plot_item.setData(x=np.array(self.xvals))
                # self.surface_plot_item.shader()['colorMap'] = pyqt_default_shader_scaled(np.array(data[:,1]))
            self.yvals.append((len(self.yvals)) * self.live_spacing + self.live_min)
            self.integral_data.append(data[:, 1])
        # print(np.array(self.xvals).shape, np.array(self.yvals).shape,np.transpose(self.integral_data))
        self.contour_image.setImage(np.array(self.integral_data))
        if self.x_axis_type == 0:
            self.contour_image.setRect(
                self.tthvals[0],
                self.yvals[0],
                self.tthvals[-1] - self.tthvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )
        elif self.x_axis_type == 1:
            self.contour_image.setRect(
                self.qvals[0],
                self.yvals[0],
                self.qvals[-1] - self.qvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )

    def update_integral_data(self, reset_z = False):
        # Save existing z values from histogram
        min_z, max_z = self.intensityBar.getLevels()
        # If the requested spacing hasn't changed, just append the new data
        if self._temp_auto_spacing == self.live_spacing:
            self.append_integral_data()
            if not reset_z: self.intensityBar.setLevels(min=min_z, max=max_z)
        else:
            # If requested spacing is divisible by old spacing, can just splice down and add back
            if self._temp_auto_spacing % self.live_spacing == 0:
                self.integral_data = self.integral_data[
                    0 :: int(self._temp_auto_spacing / self.live_spacing)
                ]
                self.yvals = self.yvals[
                    0 :: int(self._temp_auto_spacing / self.live_spacing)
                ]
                self.live_spacing = self._temp_auto_spacing
                self.live_integral_step.valueChanged.disconnect(
                    self.live_integral_step_changed
                )
                self.live_integral_step.setValue(self.live_spacing)
                self.live_integral_step.valueChanged.connect(
                    self.live_integral_step_changed
                )
                self.append_integral_data()
                if not reset_z: self.intensityBar.setLevels(min=min_z, max=max_z)
            # Otherwise, remake the dataset
            else:
                self.integral_data = []
                self.yvals = []
                self.live_spacing = self._temp_auto_spacing
                self.live_integral_step.valueChanged.disconnect(
                    self.live_integral_step_changed
                )
                self.live_integral_step.setValue(self.live_spacing)
                self.live_integral_step.valueChanged.connect(
                    self.live_integral_step_changed
                )
                self.append_integral_data()
                if not reset_z: self.intensityBar.setLevels(min=min_z, max=max_z)

        if self.viewtype_select.currentIndex() == Viewtype.Waterfall.value:
            self.update_waterfall_data()

    def update_waterfall_data(self):
        for i in self.contour_waterfall:
                self.view.removeItem(i)
        self.contour_waterfall.clear()
        if self.x_axis_type == 0:
            xvals = self.tthvals
        elif self.x_axis_type == 1:
            xvals = self.qvals
        lut = self.waterfall_cmap.getLookupTable(nPts=len(self.integral_data), mode='qcolor')
        for i in range(len(self.integral_data)):
            self.contour_waterfall.append(pg.PlotDataItem())
            self.contour_waterfall[i].setData(
                xvals,
                self.integral_data[i] + i * self.offset.value()
            )
            self.contour_waterfall[i].setPen(lut[i])
        if self.viewtype_select.currentIndex() == Viewtype.Waterfall.value:
            # see if there's an addItems
            [self.view.addItem(i) for i in self.contour_waterfall]

    def change_x_axis_type(self, axis_type):
        # 2 theta = 0, Q = 1
        self.x_axis_type = axis_type
        # self.wavelength = wavelength
        if axis_type == 0:
            self.contour_image.setRect(
                self.tthvals[0],
                self.yvals[0],
                self.tthvals[-1] - self.tthvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )
        elif axis_type == 1:
            if len(self.qvals) == 0:
                self.qvals = tth_to_q(self.tthvals, self.settings.wavelength)
            self.contour_image.setRect(
                self.qvals[0],
                self.yvals[0],
                self.qvals[-1] - self.qvals[0],
                self.yvals[-1] + self.live_spacing - self.yvals[0],
            )
        if self.viewtype_select.currentIndex() == Viewtype.Waterfall.value:
            self.update_waterfall_data()

    def update_integral_data_manual(self):
        # self.live_spacing = self.integral_step.value()
        self.append_integral_data(max=self.manual_max + 1)
        # if self._temp_auto_spacing == self.live_spacing:
        #    self.append_integral_data(max=self.manual_max)
        # else:
        #    if self._temp_auto_spacing % self.live_spacing == 0:
        #        self.integral_data = self.integral_data[]

    def reset_integral_data(self, manual=False, reset_z = False):
        self.integral_data = []
        self.xvals = []
        self.yvals = []
        if manual:
            self.live_spacing = self.integral_step.value()
            self.update_integral_list(manual=True)
        else:
            self.live_spacing = 1
            self._temp_auto_spacing = 1
            self.live_integral_step.valueChanged.disconnect(
                self.live_integral_step_changed
            )
            self.live_integral_step.setValue(self.live_spacing)
            self.live_integral_step.valueChanged.connect(
                self.live_integral_step_changed
            )
            self.update_integral_list(reset_z=reset_z)
        if self.viewtype_select.currentIndex() == Viewtype.Waterfall.value:
            self.update_waterfall_data()

    def integral_type_changed(self, evt):
        self.integral_extension = self.integral_type_dict[self.integral_types[evt]]
        # TODO: only reset list/data; keep settings
        self.reset_integral_data()

    def viewtype_changed(self, evt):
        if evt == Viewtype.Contour.value:
            self.setBackground("k")
            for i in self.contour_waterfall:
                self.view.removeItem(i)
            self.view.addItem(self.contour_image)
            self.view.autoRange()
            self.offset_label.hide()
            self.offset.hide()
            self.intensityBar.show()
        elif evt == Viewtype.Waterfall.value:
            self.setBackground("w")
            self.view.removeItem(self.contour_image)
            self.update_waterfall_data()
            self.view.autoRange()
            self.offset_label.show()
            self.offset.show()
            self.intensityBar.hide()

    def offset_changed(self):
        self.update_integral_data()

    def update_integral_live(self):
        reset = False
        if self.live_min_changed:
            self.live_min = self.live_integral_min.value()
            self.live_min_changed = False
            reset = True
        if self.live_step_changed:
            self._temp_auto_spacing = self.live_integral_step.value()
            self.live_step_changed = False
        if self.live_max_changed:
            self.live_max_lines_visible = self.live_integral_max.value()
            self.live_max_changed = False
            reset = True
        if reset:
            self.reset_integral_data()
        else:
            self.update_integral_list()

    def pause(self):
        self.timer.stop()
        for widget in self.live_controls_list:
            # widget.setEnabled(False)
            widget.hide()
        for widget in self.manual_controls_list:
            # widget.setEnabled(True)
            widget.show()
        self.live_min = self.integral_min.value()
        self.reset_integral_data(manual=True)

    def play(self):
        for widget in self.manual_controls_list:
            # widget.setEnabled(False)
            widget.hide()
        for widget in self.live_controls_list:
            # widget.setEnabled(True)
            widget.show()
        self.live_min = self.live_integral_min.value()
        self.reset_integral_data()
        self.timer.timeout.connect(self.update_integral_live)
        self.timer.start(100)

    def live_update_checkbox_changed(self):
        if self.live_update_checkbox.isChecked():
            self.play()
        else:
            self.pause()

    # Now handling actual changes on timer timeout
    def live_integral_min_changed(self):
        self.live_min_changed = True
        # self.live_min = self.live_integral_min.value()
        # There's a smarter way to do this, but for now, reset when min changes
        # self.reset_integral_data()

    def live_integral_step_changed(self):
        self.live_step_changed = True
        # self._temp_auto_spacing = self.live_integral_step.value()
        # self.update_integral_data()

    def live_integral_max_changed(self):
        self.live_max_changed = True
        # if self.live_max_lines_visible < self.live_integral_max.value():
        #    self.live_max_lines_visible = self.live_integral_max.value()
        #    #self.auto_set_spacing() #Need to have it decrease the spacing
        #    self.reset_integral_data()
        # elif self.live_max_lines_visible > self.live_integral_max.value():
        #    self.live_max_lines_visible = self.live_integral_max.value()
        #    self.reset_integral_data()

    def manual_integral_min_changed(self):
        # self.manual_min = self.integral_min.value()
        self.live_min = self.integral_min.value()
        self.reset_integral_data(manual=True)

    def manual_integral_step_changed(self):
        # self.manual_spacing = self.integral_step.value()
        self._temp_auto_spacing = self.integral_step.value()
        self.reset_integral_data(manual=True)

    def manual_integral_max_changed(self):
        # self.manual_max = self.integral_max.value()
        if self.manual_max < self.integral_max.value():
            self.manual_max = self.integral_max.value()
            self.reset_integral_data(manual=True)
        elif self.manual_max > self.integral_max.value():
            self.manual_max = self.integral_max.value()
            self.reset_integral_data(manual=True)

    def tth_line_checkbox_changed(self):
        if self.tth_line_checkbox.isChecked():
            self.tth_line.setPen(255, 255, 255, 150)
        else:
            self.tth_line.setPen(0, 0, 0, 0)


