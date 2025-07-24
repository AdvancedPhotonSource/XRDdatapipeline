import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import re
import glob
import os

import tifffile as tf

from mainUI.UI_settings import Settings, SettingsWindow
from mainUI.file_select import FileSelectWindow
from mainUI.contour import ContourView
from mainUI.file_select import FileSelectWindow
from mainUI.integrals import IntegralView
from mainUI.main_image import MainImageView
from mainUI.tabbed_area import TabbedArea

from corrections_and_maps import tth_to_q, tth_to_d, q_to_tth


class KeyPressWindow(QtWidgets.QWidget):
    def __init__(self, image_directory=".", output_directory = ".", imagecontrol=""):
        super().__init__()
        self.settings = Settings(
            image_directory, output_directory, imagecontrol, (10, 10), 0, 0, [], {}, 0, 0
        )
        self.file_select_widget = FileSelectWindow(self.settings)
        self.file_select_widget.file_selected.connect(self.update_dir)
        self.settings_widget = SettingsWindow(self.settings)
        self.settings_widget.apply_settings.connect(self.update_settings)
        # self.directory = directory
        # self.update_tiflist()
        self.menubar = pg.QtWidgets.QMenuBar()
        self.menubar.setMaximumHeight(40)
        self.filemenu = self.menubar.addMenu("&File")
        self.filemenu.addAction(
            "Load Directory and Image Control File", self.choose_dir
        )
        self.settingsmenu = self.menubar.addMenu("&Settings")
        self.settingsmenu.addAction("Open Settings...", self.show_settings)
        self.imageview = MainImageView(self.settings, parent=self)
        # self.csimview = CSimView(self, self.settings)
        self.contourview = ContourView(self, self.settings)
        self.integral_widget = IntegralView(self, self.settings)
        self.tabbed_area = TabbedArea(self, self.settings)
        self.tabbed_area.user_data_widget.update_userdata_signal.connect(self.update_user_data)
        self.tabbed_area.user_data_widget.remove_deleted_userdata_from_plot.connect(self.remove_user_data)

        # Show name of image
        # self.name_label = QtWidgets.QLabel("<span style='font-size: 12pt'>{0}</span>".format(tiflist[keylist[curr_key]][curr_pos]))
        # self.name_label = QtWidgets.QLabel()
        # self.name_label.setMinimumSize(500, 50)

        # Placing the mouseMoved() event here so it can more easily be passed to the integral widget
        self.cursor_label = QtWidgets.QLabel(
            "<span style='font-size: 12pt'>x=%0.0f,   y=%0.0f</span>,   <span style='color: red; font-size: 12pt'>2Theta=%0.1f,    Azim=%0.1f,</span>    <span style='font-size: 12pt'>Q=%0.0f,    d=%0.0f</span>"
            % (0, 0, 0, 0, 0, 0)
        )
        self.cursor_label.setMinimumSize(500, 50)
        # self.cursor_label.setStyleSheet("background-color: black") # still leaves a small break
        self.integral_cursor_label = pg.TextItem(text="tth= \nq= \nd= ", anchor=(0, 1))
        self.integral_widget.integral_view.addItem(
            self.integral_cursor_label, ignoreBounds=True
        )
        self.imageview.view.scene().sigMouseMoved.connect(self.mouseMovedImage)
        self.integral_widget.integral_view.scene().sigMouseMoved.connect(
            self.mouseMovedIntegral
        )
        # self.integral_widget.integral_view.scene().sigMouseHover.connect(self.mouseHoverIntegral)
        self.tabbed_area.contour_widget.view.scene().sigMouseMoved.connect(
            self.mouseMovedContour
        )
        self.contourview.view.scene().sigMouseMoved.connect(
            self.mouseMovedLeftContour
        )
        # self.tabbed_area.stats_widget.stats_view.scene().sigMouseMoved.connect(self.mouseMovedStats)
        self.tabbed_area.contour_widget.view.scene().sigMouseClicked.connect(
            self.mouseClickedContourChangeImage
        )
        self.contourview.view.scene().sigMouseClicked.connect(
            self.mouseClickedLeftContourChangeImage
        )
        self.tabbed_area.spottiness_widget.view.scene().sigMouseMoved.connect(
            self.mouseMovedSpottiness
        )
        self.tabbed_area.stats_widget.stats_view.scene().sigMouseMoved.connect(
            self.mouseMovedStats
        )
        # checkboxes for the vertical line and circle
        self.vLineCheckbox = QtWidgets.QCheckBox("2Th Line")
        # self.vLineCheckbox.setChecked(True)
        self.vLineCheckbox.stateChanged.connect(self.vLineCheckbox_changed)
        self.circleCheckbox = QtWidgets.QCheckBox("2Th Circle")
        # self.circleCheckbox.setChecked(True)
        self.circleCheckbox.stateChanged.connect(self.circleCheckbox_changed)
        self.linked_axes_checkbox = QtWidgets.QCheckBox("Link X axes")
        self.linked_axes_checkbox.stateChanged.connect(
            self.linked_axes_checkbox_changed
        )

        self.live_view_image_checkbox = QtWidgets.QCheckBox("Live update")
        self.live_view_image_checkbox.stateChanged.connect(
            self.live_view_image_checkbox_changed
        )
        self.x_axis_choice = QtWidgets.QComboBox()
        self.x_axis_types = ["2 Theta", "Q"]
        self.x_axis_choice.addItems(self.x_axis_types)
        self.x_axis_choice.setCurrentIndex(0)
        self.x_axis_choice.currentIndexChanged.connect(self.x_axis_changed)
        # self.tooltip = QtWidgets.QToolTip()
        # self.tooltip.isVisible()

        # win.show()
        self.layout = QtWidgets.QGridLayout()
        # addWidget(widget,row,col,row_length,col_length)
        self.layout.addWidget(self.menubar, 0, 0, 1, 11)
        self.layout.addWidget(self.imageview, 1, 0, 7, 5)
        # self.layout.addWidget(self.name_label, 8, 0, 1, 4)
        self.layout.addWidget(self.live_view_image_checkbox, 8, 4)
        self.layout.addWidget(self.cursor_label, 8, 0, 1, 4)
        self.layout.addWidget(self.imageview.predef_mask_box, 9, 0)
        self.layout.addWidget(self.imageview.mask_box, 9, 1)
        self.layout.addWidget(self.imageview.spot_mask_box, 9, 2)
        self.layout.addWidget(self.imageview.arcs_mask_box, 9, 3)
        self.layout.addWidget(self.circleCheckbox, 10, 0)
        self.layout.addWidget(self.imageview.mask_opacity_label, 10, 1)
        self.layout.addWidget(self.imageview.mask_opacity_box, 10, 2)
        # self.layout.addWidget(self.csimview, 11, 0, 3, 5)
        # self.layout.addWidget(self.contourview, 11, 0, 3, 5)
        self.contour_layout = QtWidgets.QGridLayout()
        self.contour_widget = QtWidgets.QWidget()
        self.contour_widget.setLayout(self.contour_layout)
        self.contour_layout.addWidget(self.contourview, 0, 0, 5, 6)
        self.contour_layout.addWidget(self.contourview.live_update_checkbox, 5, 0)
        self.contour_layout.addWidget(self.contourview.tth_line_checkbox, 5, 1)
        self.contour_layout.addWidget(self.contourview.integral_select, 5, 2)
        self.contour_layout.addWidget(self.contourview.live_integral_min_label, 6, 0)
        self.contour_layout.addWidget(self.contourview.live_integral_min, 6, 1)
        self.contour_layout.addWidget(self.contourview.live_integral_max_label, 6, 2)
        self.contour_layout.addWidget(self.contourview.live_integral_max, 6, 3)
        self.contour_layout.addWidget(
            self.contourview.live_integral_step_label, 6, 4
        )
        self.contour_layout.addWidget(self.contourview.live_integral_step, 6, 5)
        self.contour_layout.addWidget(self.contourview.integral_min_label, 7, 0)
        self.contour_layout.addWidget(self.contourview.integral_min, 7, 1)
        self.contour_layout.addWidget(self.contourview.integral_max_label, 7, 2)
        self.contour_layout.addWidget(self.contourview.integral_max, 7, 3)
        self.contour_layout.addWidget(self.contourview.integral_step_label, 7, 4)
        self.contour_layout.addWidget(self.contourview.integral_step, 7, 5)
        self.layout.addWidget(self.contour_widget, 11, 0, 3, 5)
        # blank space to help ameliorate formatting... should set sizes for other widgets
        # self.layout.addWidget(QtWidgets.QWidget(), 11, 0)

        self.layout.addWidget(self.integral_widget, 1, 5, 2, 7)
        self.layout.addWidget(self.integral_widget.base_integral_checkbox, 3, 5)
        self.layout.addWidget(self.integral_widget.masked_integral_checkbox, 3, 6)
        self.layout.addWidget(self.integral_widget.spotmasked_integral_checkbox, 3, 7)
        self.layout.addWidget(
            self.integral_widget.texturemasked_integral_checkbox, 3, 8
        )
        self.layout.addWidget(self.integral_widget.spots_diff_integral_checkbox, 3, 9)
        self.layout.addWidget(self.integral_widget.arcs_diff_integral_checkbox, 3, 10)
        # other
        self.layout.addWidget(self.integral_widget.integral_offset_text, 4, 5)
        self.layout.addWidget(self.integral_widget.integral_offset, 4, 6)
        self.layout.addWidget(self.integral_widget.sqrt_checkbox, 4, 7)
        self.layout.addWidget(self.x_axis_choice, 4, 8)
        self.layout.addWidget(self.vLineCheckbox, 5, 5)
        self.layout.addWidget(self.linked_axes_checkbox, 5, 6)

        self.layout.addWidget(self.tabbed_area, 6, 5, 8, 7)

        self.setLayout(self.layout)
        self.show()

        self.timer = QtCore.QTimer()

    # def start_cycling(self,tick=100):
    #    self.timer.start(tick) #interval in ms
    # def stop_cycling(self):
    #    self.timer.stop()

    def show_settings(self):
        # settings_widget = SettingsWindow(self.settings)
        self.settings_widget.update_shown_info()
        self.settings_widget.show()

    def choose_dir(self):
        self.file_select_widget.update_shown_info()
        self.file_select_widget.show()

    def update_dir(self):
        self.settings.curr_key = 0
        self.settings.curr_pos = 0
        if ".imctrl" in self.settings.imagecontrol:
            with open(self.settings.imagecontrol, "r") as infile:
                filetext = infile.read()
            matches = re.findall("wavelength:([\d\.]+)", filetext)
            self.settings.wavelength = float(matches[0])
            matches = re.findall("outChannels:([\d.]+)", filetext)
            self.settings.outChannels = int(matches[0])
            # print(matches[0])
        elif ".poni" in self.settings.imagecontrol:
            with open(self.settings.imagecontrol, "r") as infile:
                filetext = infile.read()
            matches = re.findall("Wavelength: ([\d.e+-]+)", filetext)
            self.settings.wavelength = float(matches[0]) * (10**10)
        self.update_tiflist()
        self.update_num()
        self.imageview.update_dir()
        # self.csimview.update_dir()
        self.contourview.update_dir()
        self.integral_widget.update_dir()
        self.tabbed_area.update_dir()
        # self.name_label.setText(
        self.imageview.view.setTitle(
            "<span style='font-size: 12pt'>{0}</span>".format(
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][
                    self.settings.curr_pos
                ]
            )
        )

    def update_settings(self):
        self.settings.curr_key = 0
        self.settings.curr_pos = 0
        self.update_tiflist()
        self.imageview.update_dir()
        self.integral_widget.update_dir()
        self.tabbed_area.update_dir()
        # self.name_label.setText(
        self.imageview.view.setTitle(
            "<span style='font-size: 12pt'>{0}</span>".format(
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][
                    self.settings.curr_pos
                ]
            )
        )
        matchstring = rf".*{re.escape(self.settings.curr_key)}(?P<number>\d{5}|\d{5}[_\-]\d{5})\..*"
        matches = re.match(matchstring, self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos])
        self.settings.curr_num = matches.group("number")

    def update_tiflist(self):
        # global tiflist, keylist, curr_key, curr_pos
        fulltiflist = sorted(
            glob.glob(self.settings.image_directory + "/*.tif"), key = lambda x: (len(x), x)
        )
        keylist = []
        tiflist = {}
        for tif in fulltiflist: # Probably losing sort here
            #key = tif.split("\\")[-1].split("/")[-1].split("-")[0] # grab label at start of file name, eg "Sam4". Need to work on this, as some are things like "Dewen-4"
            key = os.path.split(tif)[1]
            result = re.search(r"(\d{5})", key)
            if result is not None:
                key = key[0 : result.start(0)]
                if key not in keylist:
                    keylist.append(key)
                    tiflist[key] = []
            #initialimage = tif[0:re.search(r'(\d+)\D+$', tif).end(1)] # string from the start to the end of the last set of numbers.
            initialimage = os.path.splitext(os.path.split(tif)[1])[0]
            if initialimage not in tiflist[key]:
                tiflist[key].append(initialimage)
        self.settings.keylist = keylist
        self.settings.tiflist = tiflist
        # get size of first image, if it exists
        self.settings.image_size = self.get_image_size(
            os.path.join(
                self.settings.image_directory,
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos]+".tif"
            )
        )

    def get_image_size(self, imagename):
        image = tf.imread(imagename)
        return image.shape

    def check_for_updates(self):
        # global tiflist, keylist, curr_key, curr_pos
        # First thing saved: zero mask
        # Last things saved: integrals
        # Check integral list against tiflist
        self.update_tiflist()
        self.tabbed_area.contour_widget.update_integral_list()
        self.contourview.update_integral_list()
        last_completed = self.tabbed_area.contour_widget.integral_filelist[-1]
        # print(last_completed.split("\\")[1].split(self.tabbed_area.contour_widget.integral_extension)[0])
        last_completed = os.path.splitext(os.path.split(last_completed)[1])[0]
        # print(tiflist[keylist[curr_key]])
        completed_pos = self.settings.tiflist[
            self.settings.keylist[self.settings.curr_key]
        ].index(last_completed)
        # print(completed_pos)
        if completed_pos != self.settings.curr_pos:
            self.settings.curr_pos = completed_pos
            self.updateImages()

    def updateImages(self, z_reset=False):
        # global curr_pos
        # self.name_label.setText(
        self.imageview.view.setTitle(
            "<span style='font-size: 12pt'>{0}</span>".format(
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][
                    self.settings.curr_pos
                ]
            )
        )
        self.imageview.update_image_data(z_reset=z_reset)
        self.imageview.update_masks_data()
        self.integral_widget.update_integral_data()
        self.tabbed_area.contour_widget.horiz_line.setValue(self.settings.curr_pos)
        self.contourview.horiz_line.setValue(self.settings.curr_pos)
        self.tabbed_area.stats_widget.update_stats_data()
        self.tabbed_area.spottiness_widget.update_data()

    def update_user_data(self, data, location):
        integral_plot = self.integral_widget.integral_view
        stats_plot = self.tabbed_area.stats_widget.stats_view
        if location != "Integral":
            if data.plotitem in integral_plot.listDataItems():
                integral_plot.removeItem(data.plotitem)
        if location != "Stats":
            if data.plotitem in stats_plot.listDataItems():
                stats_plot.removeItem(data.plotitem)
        if location == "Integral":
            if data.plotitem not in integral_plot.listDataItems():
                self.integral_widget.integral_view.addItem(data.plotitem)
        elif location == "Stats":
            if data.plotitem not in stats_plot.listDataItems():
                stats_plot.addItem(data.plotitem)

    def remove_user_data(self, data):
        integral_plot = self.integral_widget.integral_view
        stats_plot = self.tabbed_area.stats_widget.stats_view
        if data.plotitem in integral_plot.listDataItems():
            integral_plot.removeItem(data.plotitem)
        if data.plotitem in stats_plot.listDataItems():
            stats_plot.removeItem(data.plotitem)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.forward()
        elif event.key() == QtCore.Qt.Key_Left:
            self.backward()
        elif event.key() == QtCore.Qt.Key_Up:
            self.prevkey()
        elif event.key() == QtCore.Qt.Key_Down:
            self.nextkey()
        elif event.key() == QtCore.Qt.Key_Space:
            self.update_tiflist()
        # elif event.key() == Qt.Key_F:
        #     self.start_cycling()
        # elif event.key() == Qt.Key_G:
        #     self.stop_cycling()
        # else:
        #    print("Only Left, Right, Up, Down and Space keys are functional!")

    def update_num(self):
        matchstring = rf".*{re.escape(self.settings.keylist[self.settings.curr_key])}" + r"(?P<number>\d{5}[_\-]\d{5}|\d{5})"
        matches = re.match(matchstring, self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos])
        self.settings.curr_num = matches.group("number")

    def forward(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos += 1
        self.settings.curr_pos = self.settings.curr_pos % len(
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]]
        )
        self.update_num()
        self.updateImages()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def backward(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos -= 1
        self.settings.curr_pos = self.settings.curr_pos % len(
            self.settings.tiflist[self.settings.keylist[self.settings.curr_key]]
        )
        self.update_num()
        self.updateImages()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def prevkey(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos = 0
        self.settings.curr_key -= 1
        self.settings.curr_key = self.settings.curr_key % len(self.settings.keylist)
        self.update_num()
        self.updateImages(z_reset=True)
        self.tabbed_area.contour_widget.reset_integral_data(reset_z=True)
        self.contourview.reset_integral_data(reset_z = True)
        self.tabbed_area.csim_widget.update_data()
        # self.csimview.update_data()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def nextkey(self):
        # global tiflist, keylist, curr_key, curr_pos
        self.settings.curr_pos = 0
        self.settings.curr_key += 1
        self.settings.curr_key = self.settings.curr_key % len(self.settings.keylist)
        self.update_num()
        self.updateImages(z_reset=True)
        self.tabbed_area.contour_widget.reset_integral_data(reset_z=True)
        self.contourview.reset_integral_data(reset_z = True)
        self.tabbed_area.csim_widget.update_data()
        # self.csimview.update_data()
        # self.setWindowTitle(tiflist[keylist[curr_key]][curr_pos])

    def mouseMovedImage(self, evt):
        pos = evt
        if self.imageview.view.sceneBoundingRect().contains(pos):
            mousePoint = self.imageview.view.vb.mapSceneToView(pos)
            x_val = int(mousePoint.x())
            y_val = int(mousePoint.y())
            if (
                x_val > 0
                and x_val < self.imageview.image_data.shape[1]
                and y_val > 0
                and y_val < self.imageview.image_data.shape[0]
            ):
                tth = self.imageview.tth_map[y_val, x_val]
                azim = self.imageview.azim_map[y_val, x_val]
                if self.settings.wavelength != 0:
                    # Q = 4*np.pi*np.sin(tth/2 * np.pi/180) / self.wavelength #inverse angstroms
                    Q = tth_to_q(tth, self.settings.wavelength)
                    # d = self.wavelength / (2 * np.sin(tth/2 * np.pi/180)) #angstroms
                    d = tth_to_d(tth, self.settings.wavelength)
                else:
                    Q = 0
                    d = 0
                # calc intensity of underlying image
                z = self.imageview.image_data[int(mousePoint.y()), int(mousePoint.x())]
                self.cursor_label.setText(
                    "<span style='font-size: 12pt'>x=%0.0f,   y=%0.0f,    z=%0.0f</span>,   <span style='color: red; font-size: 12pt'>2Theta=%0.2f,    Azim=%0.1f,</span>    <span style='font-size: 12pt'>Q=%0.2f,    d=%0.2f</span>"
                    % (mousePoint.x(), mousePoint.y(), z, tth, azim, Q, d)
                )
                if self.vLineCheckbox.isChecked():
                    integral_point = mousePoint
                    integral_point.setY(
                        self.integral_widget.integral_view.getAxis("left").range[0]
                    )
                    if self.x_axis_choice.currentIndex() == 0:
                        self.integral_widget.vLine.setPos(tth)
                        integral_point.setX(tth)
                        self.integral_cursor_label.setPos(integral_point)
                        self.tabbed_area.spottiness_widget.vLine.setPos(tth)
                        self.tabbed_area.contour_widget.tth_line.setPos(tth)
                        self.tabbed_area.stats_widget.vLine.setPos(tth)
                        self.contourview.tth_line.setPos(tth)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = tth_to_q(tth, self.settings.wavelength)
                        self.integral_widget.vLine.setPos(Q)
                        self.tabbed_area.spottiness_widget.vLine.setPos(Q)
                        self.tabbed_area.contour_widget.tth_line.setPos(Q)
                        self.tabbed_area.stats_widget.vLine.setPos(Q)
                        self.contourview.tth_line.setPos(Q)
                        integral_point.setX(Q)
                        self.integral_cursor_label.setPos(integral_point)
                    self.integral_cursor_label.setText(
                        "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                    )
                if self.circleCheckbox.isChecked():
                    self.imageview.update_tth_circle(tth)
                # if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
                #     if self.x_axis_choice.currentIndex() == 0:
                #         self.tabbed_area.contour_widget.tth_line.setPos(tth)
                #     elif self.x_axis_choice.currentIndex() == 1:
                #         Q = tth_to_q(tth, self.settings.wavelength)
                #         self.tabbed_area.contour_widget.tth_line.setPos(Q)
                # if self.tabbed_area.stats_line_checkbox.isChecked():
                #    self.tabbed_area.stats_widget.stats_line.setPos(tth)

    def mouseMovedIntegral(self, evt):
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.stats_line_checkbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
        pos = evt
        if self.integral_widget.integral_view.sceneBoundingRect().contains(pos):
            mousePoint = self.integral_widget.integral_view.vb.mapSceneToView(pos)
            if self.vLineCheckbox.isChecked():
                self.integral_cursor_label.setPos(mousePoint)
                if self.x_axis_choice.currentIndex() == 0:
                    tth = mousePoint.x()
                    self.integral_widget.vLine.setPos(tth)
                    self.tabbed_area.spottiness_widget.vLine.setPos(tth)
                    self.tabbed_area.stats_widget.vLine.setPos(tth)
                    # Q = 4*np.pi*np.sin(tth/2 * np.pi/180) / self.wavelength
                    # d = self.wavelength / (2 * np.sin(tth/2 * np.pi/180))
                    Q = tth_to_q(tth, self.settings.wavelength)
                elif self.x_axis_choice.currentIndex() == 1:
                    Q = mousePoint.x()
                    self.integral_widget.vLine.setPos(Q)
                    self.tabbed_area.spottiness_widget.vLine.setPos(Q)
                    self.tabbed_area.stats_widget.vLine.setPos(Q)
                    tth = q_to_tth(Q, self.settings.wavelength)
                # axes for both are swapped at the same time, so it can still use mousePoint.x()
                self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
                self.contourview.tth_line.setPos(mousePoint.x())
                d = tth_to_d(tth, self.settings.wavelength)
                self.integral_cursor_label.setText(
                    "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                )
            if self.circleCheckbox.isChecked():
                if self.x_axis_choice.currentIndex() == 0:
                    self.imageview.update_tth_circle(mousePoint.x())
                elif self.x_axis_choice.currentIndex() == 1:
                    self.imageview.update_tth_circle(
                        q_to_tth(mousePoint.x(), self.settings.wavelength)
                    )
            # if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
            #     # axes for both are swapped at the same time, so it can still use mousePoint.x()
            #     self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
            # if self.tabbed_area.stats_line_checkbox.isChecked():
            #    self.tabbed_area.stats_widget.stats_line.setPos(mousePoint.x())
            # self.tooltip.showText(mousePoint,"test")
            # print(mousePoint)
            # pos.mapToGlobal() does not exist
            # evt.screenPos() does not exist
            # self.hoverpos = pos
            # QtWidgets.QToolTip.showText(self.hoverpos.toPoint(),"test")
            # self.integral_widget.integral_view.setToolTip("test")

    # def mouseHoverIntegral(self,evt):
    #    print(self.hoverpos.toPoint())
    #    #print(self.hoverpos.toPoint(),self.hoverpos.toPoint().screenPos())
    #    QtWidgets.QToolTip.showText(self.hoverpos.toPoint(),"test")

    def mouseMovedContour(self, evt):
        if (
            self.vLineCheckbox.isChecked()
            or self.circleCheckbox.isChecked()
            or self.tabbed_area.contour_widget.tth_line_checkbox.isChecked()
        ):
            pos = evt
            if self.tabbed_area.contour_widget.view.sceneBoundingRect().contains(pos):
                mousePoint = self.tabbed_area.contour_widget.view.vb.mapSceneToView(pos)
                integral_point = mousePoint
                # integral_point.setY(self.integral_widget.integral_view.getAxis("left").range[1]*.85) #near top
                # integral_point.setY(0)
                # axis = self.integral_widget.integral_view.getAxis("left")
                # integral_point.setY(axis.range[0] + (axis.range[1]-axis.range[0])*.05)
                integral_point.setY(
                    self.integral_widget.integral_view.getAxis("left").range[0]
                )
                self.integral_cursor_label.setPos(integral_point)
                if self.vLineCheckbox.isChecked():
                    self.integral_widget.vLine.setPos(mousePoint.x())
                    self.tabbed_area.spottiness_widget.vLine.setPos(mousePoint.x())
                    self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
                    self.contourview.tth_line.setPos(mousePoint.x())
                    if self.x_axis_choice.currentIndex() == 0:
                        tth = mousePoint.x()
                        Q = tth_to_q(tth, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = mousePoint.x()
                        tth = q_to_tth(Q, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    self.integral_cursor_label.setText(
                        "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                    )
                if self.circleCheckbox.isChecked():
                    if self.x_axis_choice.currentIndex() == 0:
                        self.imageview.update_tth_circle(mousePoint.x())
                    elif self.x_axis_choice.currentIndex() == 1:
                        self.imageview.update_tth_circle(
                            q_to_tth(mousePoint.x(), self.settings.wavelength)
                        )
                # if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
                #     self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
    
    def mouseMovedLeftContour(self, evt):
        if (
            self.vLineCheckbox.isChecked()
            or self.circleCheckbox.isChecked()
            or self.contourview.tth_line_checkbox.isChecked()
        ):
            pos = evt
            if self.contourview.view.sceneBoundingRect().contains(pos):
                mousePoint = self.contourview.view.vb.mapSceneToView(pos)
                integral_point = mousePoint
                # integral_point.setY(self.integral_widget.integral_view.getAxis("left").range[1]*.85) #near top
                # integral_point.setY(0)
                # axis = self.integral_widget.integral_view.getAxis("left")
                # integral_point.setY(axis.range[0] + (axis.range[1]-axis.range[0])*.05)
                integral_point.setY(
                    self.integral_widget.integral_view.getAxis("left").range[0]
                )
                self.integral_cursor_label.setPos(integral_point)
                if self.vLineCheckbox.isChecked():
                    self.integral_widget.vLine.setPos(mousePoint.x())
                    self.tabbed_area.spottiness_widget.vLine.setPos(mousePoint.x())
                    self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
                    self.tabbed_area.stats_widget.vLine.setPos(mousePoint.x())
                    self.contourview.tth_line.setPos(mousePoint.x())
                    if self.x_axis_choice.currentIndex() == 0:
                        tth = mousePoint.x()
                        Q = tth_to_q(tth, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    elif self.x_axis_choice.currentIndex() == 1:
                        Q = mousePoint.x()
                        tth = q_to_tth(Q, self.settings.wavelength)
                        d = tth_to_d(tth, self.settings.wavelength)
                    self.integral_cursor_label.setText(
                        "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                    )
                if self.circleCheckbox.isChecked():
                    if self.x_axis_choice.currentIndex() == 0:
                        self.imageview.update_tth_circle(mousePoint.x())
                    elif self.x_axis_choice.currentIndex() == 1:
                        self.imageview.update_tth_circle(
                            q_to_tth(mousePoint.x(), self.settings.wavelength)
                        )
                # if self.contourview.tth_line_checkbox.isChecked():
                #     self.contourview.tth_line.setPos(mousePoint.x())
    
    def mouseMovedSpottiness(self, evt):
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.stats_line_checkbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked():
        # if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked() or self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
        pos = evt
        if self.tabbed_area.spottiness_widget.view.sceneBoundingRect().contains(pos):
            mousePoint = self.tabbed_area.spottiness_widget.view.vb.mapSceneToView(pos)
            integral_point = mousePoint
            integral_point.setY(
                self.integral_widget.integral_view.getAxis("left").range[0]
            )
            if self.vLineCheckbox.isChecked():
                self.integral_cursor_label.setPos(integral_point)
                if self.x_axis_choice.currentIndex() == 0:
                    tth = mousePoint.x()
                    self.integral_widget.vLine.setPos(tth)
                    self.tabbed_area.spottiness_widget.vLine.setPos(tth)
                    # Q = 4*np.pi*np.sin(tth/2 * np.pi/180) / self.wavelength
                    # d = self.wavelength / (2 * np.sin(tth/2 * np.pi/180))
                    Q = tth_to_q(tth, self.settings.wavelength)
                elif self.x_axis_choice.currentIndex() == 1:
                    Q = mousePoint.x()
                    self.integral_widget.vLine.setPos(Q)
                    self.tabbed_area.spottiness_widget.vLine.setPos(Q)
                    tth = q_to_tth(Q, self.settings.wavelength)
                d = tth_to_d(tth, self.settings.wavelength)
                self.integral_cursor_label.setText(
                    "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                )
                # axes for both are swapped at the same time, so it can still use mousePoint.x()
                self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())
                self.contourview.tth_line.setPos(mousePoint.x())
            if self.circleCheckbox.isChecked():
                if self.x_axis_choice.currentIndex() == 0:
                    self.imageview.update_tth_circle(mousePoint.x())
                elif self.x_axis_choice.currentIndex() == 1:
                    self.imageview.update_tth_circle(
                        q_to_tth(mousePoint.x(), self.settings.wavelength)
                    )
            # if self.tabbed_area.contour_widget.tth_line_checkbox.isChecked():
            #     # axes for both are swapped at the same time, so it can still use mousePoint.x()
            #     self.tabbed_area.contour_widget.tth_line.setPos(mousePoint.x())

    def mouseMovedStats(self,evt):
        if self.vLineCheckbox.isChecked() or self.circleCheckbox.isChecked():
            pos = evt
            if self.tabbed_area.stats_widget.stats_view.sceneBoundingRect().contains(pos):
                mousePoint = self.tabbed_area.stats_widget.stats_view.vb.mapSceneToView(pos)
                integral_point = mousePoint
                integral_point.setY(
                    self.integral_widget.integral_view.getAxis("left").range[0]
                )
                if self.x_axis_choice.currentIndex() == 0:
                    tth = mousePoint.x()
                    Q = tth_to_q(tth, self.settings.wavelength)
                elif self.x_axis_choice.currentIndex() == 1:
                    Q = mousePoint.x()
                    tth = q_to_tth(Q, self.settings.wavelength)
                d = tth_to_d(tth, self.settings.wavelength)
                if self.vLineCheckbox.isChecked():
                    if self.x_axis_choice.currentIndex() == 0:
                        self.integral_cursor_label.setPos(integral_point)
                        self.integral_widget.vLine.setPos(tth)
                        self.tabbed_area.stats_widget.vLine.setPos(tth)
                        self.contourview.tth_line.setPos(tth)
                    elif self.x_axis_choice.currentIndex() == 1:
                        self.integral_cursor_label.setPos(integral_point)
                        self.integral_widget.vLine.setPos(Q)
                        self.tabbed_area.stats_widget.vLine.setPos(Q)
                        self.contourview.tth_line.setPos(Q)
                    self.integral_cursor_label.setText(
                        "tth={0:0.2f}\nQ={1:0.2f}\nd={2:0.2f}".format(tth, Q, d)
                    )
                if self.circleCheckbox.isChecked():
                    self.imageview.update_tth_circle(tth)

    def mouseClickedContourChangeImage(self, evt):
        # global tiflist, keylist, curr_key, curr_pos
        # evt.button() results = {1: left click, 2: right click, 4: middle click}
        # print(evt.button())
        # If left click
        # Can also check for double-click (evt.double() == True)
        # no longer outputting same evt.button() results
        # if evt.button() == 1:
        #     print("Recognized left click")
        #     pos = int(self.tabbed_area.contour_widget.view.vb.mapSceneToView(evt.scenePos()).y())
        #     if (pos >= 0) and (pos >= self.tabbed_area.contour_widget.view.getAxis("left").range[0]) and (pos < len(self.settings.tiflist[self.settings.keylist[self.settings.curr_key]])):
        #         if self.timer.isActive():
        #             self.pause()
        #         self.curr_pos = pos
        #         print(pos)
        #         self.tabbed_area.contour_widget.horiz_line.setValue(pos)
        #         self.updateImages()
        if evt.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            pos = int(
                self.tabbed_area.contour_widget.view.vb.mapSceneToView(
                    evt.scenePos()
                ).y()
            )
            if (
                (pos >= 0)
                and (
                    pos >= self.tabbed_area.contour_widget.view.getAxis("left").range[0]
                )
                and (
                    pos
                    < len(
                        self.settings.tiflist[
                            self.settings.keylist[self.settings.curr_key]
                        ]
                    )
                )
            ):
                if self.timer.isActive():
                    self.pause()
                self.settings.curr_pos = pos
                self.tabbed_area.contour_widget.horiz_line.setValue(pos)
                self.contourview.horiz_line.setValue(pos)
                self.updateImages()
    
    def mouseClickedLeftContourChangeImage(self, evt):
        if evt.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            pos = int(
                self.contourview.view.vb.mapSceneToView(
                    evt.scenePos()
                ).y()
            )
            if (
                (pos >= 0)
                and (
                    pos >= self.contourview.view.getAxis("left").range[0]
                )
                and (
                    pos
                    < len(
                        self.settings.tiflist[
                            self.settings.keylist[self.settings.curr_key]
                        ]
                    )
                )
            ):
                if self.timer.isActive():
                    self.pause()
                self.settings.curr_pos = pos
                self.tabbed_area.contour_widget.horiz_line.setValue(pos)
                self.contourview.horiz_line.setValue(pos)
                self.updateImages()

    def vLineCheckbox_changed(self, evt):
        if self.vLineCheckbox.isChecked():
            self.integral_widget.vLine.setPen(0, 0, 0, 150)
        else:
            self.integral_widget.vLine.setPen(0, 0, 0, 0)

    def circleCheckbox_changed(self, evt):
        if not self.circleCheckbox.isChecked():
            # self.imageview.tth_circle_RGBA[:,:,3] = 0
            self.imageview.tth_circle_data.set_opacity(0)
            self.imageview.tth_circle.updateImage(
                self.imageview.tth_circle_data.full_data
            )
        else:
            self.imageview.tth_circle_data.set_opacity(0.5)
            self.imageview.tth_circle.updateImage(
                self.imageview.tth_circle_data.full_data
            )

    def linked_axes_checkbox_changed(self, evt):
        if self.linked_axes_checkbox.isChecked():
            self.tabbed_area.contour_widget.view.setXLink(
                self.integral_widget.integral_view.getViewBox()
            )
            self.tabbed_area.spottiness_widget.view.setXLink(
                self.integral_widget.integral_view.getViewBox()
            )
            self.tabbed_area.stats_widget.stats_view.setXLink(
                self.integral_widget.integral_view.getViewBox()
            )
        else:
            # self.tabbed_area.contour_widget.view.getViewBox().linkView(pg.ViewBox.XAxis,None)
            self.tabbed_area.contour_widget.view.setXLink(None)
            self.tabbed_area.spottiness_widget.view.setXLink(None)
            self.tabbed_area.stats_widget.stats_view.setXLink(None)

    def x_axis_changed(self, evt):
        # Update integrals, contour graph
        # print(self.x_axis_choice.currentIndex())
        # print(evt)
        # self.integral_widget.change_x_axis_type(evt,self.settings.wavelength)
        # self.tabbed_area.contour_widget.change_x_axis_type(evt,self.settings.wavelength)
        self.integral_widget.change_x_axis_type(evt)
        self.tabbed_area.contour_widget.change_x_axis_type(evt)
        self.contourview.change_x_axis_type(evt)
        self.tabbed_area.spottiness_widget.change_x_axis_type(evt)
        self.tabbed_area.stats_widget.change_x_axis_type(evt)

    def live_view_image_checkbox_changed(self, evt):
        if self.live_view_image_checkbox.isChecked():
            self.play()
        else:
            self.pause()

    def play(self):
        # No need for a queue, just to periodically check for updates
        self.timer.timeout.connect(self.check_for_updates)
        self.timer.start(100)

    def pause(self):
        if self.live_view_image_checkbox.isChecked():
            self.live_view_image_checkbox.stateChanged.disconnect(
                self.live_view_image_checkbox_changed
            )
            self.live_view_image_checkbox.setChecked(False)
            self.live_view_image_checkbox.stateChanged.connect(
                self.live_view_image_checkbox_changed
            )
        self.timer.stop()


