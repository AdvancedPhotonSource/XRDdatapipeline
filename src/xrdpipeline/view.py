import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
import glob
import re
import numpy as np
from skimage.io import imread
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")


class ImageAndMask(QtWidgets.QWidget):
    def __init__(self, cmap, is_first_image=False):
        super().__init__()
        self.imview = pg.ImageView()
        self.immask = pg.ImageItem(levels=None)
        self.imview.view.addItem(self.immask)
        self.imview.setColorMap(cmap)
        self.imview.view.invertY(False)
        self.raw_mask_data = []
        self.is_first_image = is_first_image

        self.title = QtWidgets.QLineEdit()

        self.imdir_label = QtWidgets.QLabel("Image subdir:")
        self.image_dir = QtWidgets.QLineEdit()
        self.imext_label = QtWidgets.QLabel("Image extension:")
        self.image_ext = QtWidgets.QLineEdit()
        self.maskdir_label = QtWidgets.QLabel("Mask subdir:")
        self.mask_dir = QtWidgets.QLineEdit()
        self.mask_dir.setText("masks" + os.sep)
        self.maskext_label = QtWidgets.QLabel("Mask extension:")
        self.mask_ext = QtWidgets.QLineEdit()

        # z options:
        self.z_options_label = QtWidgets.QLabel("Intensity options:")
        # link to first image
        self.link_to_first_image_checkbox = QtWidgets.QCheckBox(
            "Link intensity to top-left image"
        )
        # default to (0,99.9th)
        self.image_scaling_checkbox = QtWidgets.QCheckBox("Detector image")
        self.image_scaling_checkbox.setChecked(True)
        # default to (-10000,10000)
        self.gradient_scaling_checkbox = QtWidgets.QCheckBox("Gradient image")

        self.im_layout = QtWidgets.QGridLayout()
        self.im_layout.addWidget(self.title, 0, 0, 1, 8)
        self.im_layout.addWidget(self.imdir_label, 1, 0, 1, 1)
        self.im_layout.addWidget(self.image_dir, 1, 1, 1, 1)
        self.im_layout.addWidget(self.imext_label, 1, 2, 1, 1)
        self.im_layout.addWidget(self.image_ext, 1, 3, 1, 1)
        self.im_layout.addWidget(self.maskdir_label, 1, 4, 1, 1)
        self.im_layout.addWidget(self.mask_dir, 1, 5, 1, 1)
        self.im_layout.addWidget(self.maskext_label, 1, 6, 1, 1)
        self.im_layout.addWidget(self.mask_ext, 1, 7, 1, 1)
        self.im_layout.addWidget(self.z_options_label, 2, 0, 1, 1)
        if not self.is_first_image:
            self.im_layout.addWidget(self.link_to_first_image_checkbox, 2, 1, 1, 1)
        self.im_layout.addWidget(self.image_scaling_checkbox, 2, 2, 1, 1)
        self.im_layout.addWidget(self.gradient_scaling_checkbox, 2, 3, 1, 1)
        self.im_layout.addWidget(self.imview, 3, 0, 8, 8)

        self.setLayout(self.im_layout)

        # newer versions of Qt swapping to checkStateChanged
        self.image_scaling_checkbox.stateChanged.connect(
            self.image_scaling_checkbox_changed
        )
        self.gradient_scaling_checkbox.stateChanged.connect(
            self.gradient_scaling_checkbox_changed
        )

    def hide_config(self):
        self.im_layout.removeWidget(self.imdir_label)
        self.im_layout.removeWidget(self.image_dir)
        self.im_layout.removeWidget(self.imext_label)
        self.im_layout.removeWidget(self.image_ext)
        self.im_layout.removeWidget(self.maskdir_label)
        self.im_layout.removeWidget(self.mask_dir)
        self.im_layout.removeWidget(self.maskext_label)
        self.im_layout.removeWidget(self.mask_ext)
        self.im_layout.removeWidget(self.z_options_label)
        if not self.is_first_image:
            self.im_layout.removeWidget(self.link_to_first_image_checkbox)
        self.im_layout.removeWidget(self.image_scaling_checkbox)
        self.im_layout.removeWidget(self.gradient_scaling_checkbox)

    def show_config(self):
        self.im_layout.addWidget(self.imdir_label, 1, 0, 1, 1)
        self.im_layout.addWidget(self.image_dir, 1, 1, 1, 1)
        self.im_layout.addWidget(self.imext_label, 1, 2, 1, 1)
        self.im_layout.addWidget(self.image_ext, 1, 3, 1, 1)
        self.im_layout.addWidget(self.maskdir_label, 1, 4, 1, 1)
        self.im_layout.addWidget(self.mask_dir, 1, 5, 1, 1)
        self.im_layout.addWidget(self.maskext_label, 1, 6, 1, 1)
        self.im_layout.addWidget(self.mask_ext, 1, 7, 1, 1)
        self.im_layout.addWidget(self.z_options_label, 2, 0, 1, 1)
        if not self.is_first_image:
            self.im_layout.addWidget(self.link_to_first_image_checkbox, 2, 1, 1, 1)
        self.im_layout.addWidget(self.image_scaling_checkbox, 2, 2, 1, 1)
        self.im_layout.addWidget(self.gradient_scaling_checkbox, 2, 3, 1, 1)

    def image_scaling_checkbox_changed(self, state):
        # 0 = unchecked, 1 = partially checked (unavailable), 2 = checked
        if state:
            self.gradient_scaling_checkbox.setChecked(False)

    def gradient_scaling_checkbox_changed(self, state):
        if state:
            self.image_scaling_checkbox.setChecked(False)

    def resetHistoRange(self, maxval):
        if self.link_to_first_image_checkbox.isChecked():
            self.imview.getHistogramWidget().setLevels(min=0, max=maxval)
        elif self.image_scaling_checkbox.isChecked():
            maxval = np.percentile(self.imview.imageItem.image, 99.9)
            self.imview.getHistogramWidget().setLevels(min=0, max=maxval)
        elif self.gradient_scaling_checkbox.isChecked():
            self.imview.getHistogramWidget().setLevels(min=-10000, max=10000)


class FileSelect(QtWidgets.QWidget):
    file_selected = pg.QtCore.Signal(str)

    def __init__(self, label, default_text=None, isdir=False, startdir=".", ext=None):
        super().__init__()
        self.file_select_button = QtWidgets.QPushButton(label)
        self.file_name = QtWidgets.QLabel(default_text)
        self.isdir = isdir
        self.startdir = startdir
        self.ext = ext
        # self.file_selected = pg.QtCore.pyqtSignal(str)

        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.file_select_button, 0, 0)
        self.layout.addWidget(self.file_name, 0, 1, 1, 3)
        self.setLayout(self.layout)

        self.file_select_button.released.connect(self.select_file)

    def select_file(self):
        if self.isdir:
            location = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Select Directory"
            )
            self.file_name.setText(location)
            self.file_selected.emit(location)
        else:
            location = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select File", self.startdir, self.ext
            )
            self.file_name.setText(location[0])
            self.file_selected.emit(location[0])


class KeyPressWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # bookkeeping
        self.w = QtWidgets.QWidget()
        self.setCentralWidget(self.w)
        self.image_directory_widget = FileSelect(
            "Image Directory:",
            default_text=".",
            isdir=True,
        )
        self.image_directory = "."
        self.output_directory_widget = FileSelect(
            "Output Directory:",
            default_text=".",
            isdir=True,
        )
        self.output_directory = "."
        self.hide_config_button = QtWidgets.QPushButton("Hide Config")
        self.image_name = QtWidgets.QLabel("")
        self.tiflist = []
        self.keylist = []
        self.curr_key = 0
        self.curr_pos = 0

        # visuals
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)

        # image panels
        self.image_1 = ImageAndMask(self.cmap, is_first_image=True)
        self.image_2 = ImageAndMask(self.cmap)
        self.image_3 = ImageAndMask(self.cmap)
        self.image_4 = ImageAndMask(self.cmap)
        self.image_2.imview.getView().setXLink(self.image_1.imview.getView())
        self.image_2.imview.getView().setYLink(self.image_1.imview.getView())
        self.image_3.imview.getView().setXLink(self.image_1.imview.getView())
        self.image_3.imview.getView().setYLink(self.image_1.imview.getView())
        self.image_4.imview.getView().setXLink(self.image_1.imview.getView())
        self.image_4.imview.getView().setYLink(self.image_1.imview.getView())

        # layout
        self.win_layout = QtWidgets.QGridLayout()
        self.w.setLayout(self.win_layout)
        self.topbar_layout = QtWidgets.QHBoxLayout()
        self.topbar_layout.addWidget(self.image_directory_widget, 2)
        self.topbar_layout.addWidget(self.output_directory_widget, 2)
        self.topbar_layout.addWidget(self.image_name, 2)
        self.topbar_layout.addWidget(self.hide_config_button, 2)
        self.win_layout.addLayout(self.topbar_layout, 0, 0, 1, 10)
        self.win_layout.addWidget(self.image_1, 1, 0, 5, 5)
        self.win_layout.addWidget(self.image_2, 6, 0, 5, 5)
        self.win_layout.addWidget(self.image_3, 1, 5, 5, 5)
        self.win_layout.addWidget(self.image_4, 6, 5, 5, 5)

        # signals
        self.image_directory_widget.file_selected.connect(self.image_directory_changed)
        self.output_directory_widget.file_selected.connect(self.output_directory_changed)
        self.hide_config_button.released.connect(self.config_button_pressed)

        self.image_1.image_dir.editingFinished.connect(self.updateImages)
        self.image_1.image_ext.editingFinished.connect(self.updateImages)
        self.image_1.mask_dir.editingFinished.connect(self.updateImages)
        self.image_1.mask_ext.editingFinished.connect(self.updateImages)
        self.image_2.image_dir.editingFinished.connect(self.updateImages)
        self.image_2.image_ext.editingFinished.connect(self.updateImages)
        self.image_2.mask_dir.editingFinished.connect(self.updateImages)
        self.image_2.mask_ext.editingFinished.connect(self.updateImages)
        self.image_3.image_dir.editingFinished.connect(self.updateImages)
        self.image_3.image_ext.editingFinished.connect(self.updateImages)
        self.image_3.mask_dir.editingFinished.connect(self.updateImages)
        self.image_3.mask_ext.editingFinished.connect(self.updateImages)
        self.image_4.image_dir.editingFinished.connect(self.updateImages)
        self.image_4.image_ext.editingFinished.connect(self.updateImages)
        self.image_4.mask_dir.editingFinished.connect(self.updateImages)
        self.image_4.mask_ext.editingFinished.connect(self.updateImages)

        self.image_1.imview.getHistogramWidget().sigLevelChangeFinished.connect(
            self.update_other_histo_ranges
        )
        self.image_2.link_to_first_image_checkbox.stateChanged.connect(
            self.zlink_2_to_1
        )
        self.image_3.link_to_first_image_checkbox.stateChanged.connect(
            self.zlink_3_to_1
        )
        self.image_4.link_to_first_image_checkbox.stateChanged.connect(
            self.zlink_4_to_1
        )

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

    def update_tiflist(self, reset=False):
        print(self.image_directory)
        self.fulltiflist = glob.glob(self.image_directory + os.sep + "*.tif")
        self.keylist = []
        self.tiflist = {}
        for tif in self.fulltiflist:
            short = os.path.split(tif)[1]
            # grab label at start of file name, eg "Sam4". Need to work on this, as some are things like "Dewen-4"
            key = short.split("-")[0]
            if key not in self.keylist:
                self.keylist.append(key)
                self.tiflist[key] = []
            initialimage = short[
                0 : re.search(r"(\d+)\D+$", short).end(1)
            ]  # string from the start to the end of the last set of numbers.
            if initialimage not in self.tiflist[key]:
                self.tiflist[key].append(initialimage)

        # tiflist.sort(key=lambda x: os.path.getctime(x), reverse=False)
        print("All TIF images in the working directory reloaded to the list")
        if reset:
            self.curr_key = 0
            self.curr_pos = 0
            self.updateImages()

    def updateImages(self, autoScaleRange=True, resetHistoRange=False):
        # base_image = self.findImage()
        # azim_grad_image = findImage(subdir="grads\\",ext="_azim_grad_2")
        # radial_grad_image = findImage(subdir="grads\\",ext="_radial_grad_2")
        image_1_data = self.findImage(
            subdir=self.image_1.image_dir.text(), ext=self.image_1.image_ext.text()
        )
        image_2_data = self.findImage(
            subdir=self.image_2.image_dir.text(), ext=self.image_2.image_ext.text()
        )
        image_3_data = self.findImage(
            subdir=self.image_3.image_dir.text(), ext=self.image_3.image_ext.text()
        )
        image_4_data = self.findImage(
            subdir=self.image_4.image_dir.text(), ext=self.image_4.image_ext.text()
        )
        self.image_1.imview.setImage(
            image_1_data,
            autoRange=autoScaleRange,
            autoLevels=autoScaleRange,
            autoHistogramRange=autoScaleRange,
        )
        self.image_2.imview.setImage(
            image_2_data,
            autoRange=autoScaleRange,
            autoLevels=autoScaleRange,
            autoHistogramRange=autoScaleRange,
        )
        self.image_3.imview.setImage(
            image_3_data,
            autoRange=autoScaleRange,
            autoLevels=autoScaleRange,
            autoHistogramRange=autoScaleRange,
        )
        self.image_4.imview.setImage(
            image_4_data,
            autoRange=autoScaleRange,
            autoLevels=autoScaleRange,
            autoHistogramRange=autoScaleRange,
        )

        self.addMaskImage(
            ext=self.image_1.mask_ext.text(), image_item=self.image_1.immask
        )
        self.addMaskImage(
            ext=self.image_2.mask_ext.text(), image_item=self.image_2.immask
        )
        self.addMaskImage(
            ext=self.image_3.mask_ext.text(), image_item=self.image_3.immask
        )
        self.addMaskImage(
            ext=self.image_4.mask_ext.text(), image_item=self.image_4.immask
        )
        # addMaskImage(ext="_qwidth_arc",image_item=imv2_mask)
        # imv3.setImage(findImage(ext="_arcsclosed"))
        # imv3.setImage(radial_grad_image,autoRange=autoScaleRange,autoLevels=autoScaleRange,autoHistogramRange=autoScaleRange)
        # addMaskImage(ext="_qgrad_arc",image_item=imv3_mask)
        # imv4.setImage(findImage(ext="_spotsclosed"))
        # imv4.setImage(azim_grad_image,autoRange=autoScaleRange,autoLevels=autoScaleRange,autoHistogramRange=autoScaleRange)
        # addMaskImage(ext="_qgradarcs0p1",image_item=imv4_mask)

        if resetHistoRange:
            maxval = np.percentile(image_1_data, 99.9)
            # self.image_1.imview.getHistogramWidget().setLevels(min=0,max=maxval)
            # if self.image_2.link_to_first_image_checkbox.isChecked():
            #     self.image_2.imview.getHistogramWidget().setLevels(min=0,max=maxval)
            # if self.image_3.link_to_first_image_checkbox.isChecked():
            #     self.image_3.imview.getHistogramWidget().setLevels(min=0,max=maxval)
            # if self.image_4.link_to_first_image_checkbox.isChecked():
            #     self.image_4.imview.getHistogramWidget().setLevels(min=0,max=maxval)
            self.image_1.resetHistoRange(maxval)
            self.image_2.resetHistoRange(maxval)
            self.image_3.resetHistoRange(maxval)
            self.image_4.resetHistoRange(maxval)

    def findImage(self, subdir=".", ext="", except_shape=(2880, 2880)):
        if (subdir == ".") or (subdir == ""):
            try:
                im = imread(
                    os.path.join(
                        self.image_directory,
                        subdir,
                        self.tiflist[self.keylist[self.curr_key]][self.curr_pos] + ext + ".tif"
                    )
                )
            except:
                im = np.zeros(except_shape)
        else:
            try:
                im = imread(
                    os.path.join(
                        self.output_directory,
                        subdir,
                        self.tiflist[self.keylist[self.curr_key]][self.curr_pos] + ext + ".tif"
                    )
                )
            except:
                im = np.zeros(except_shape)
        return im

    def addMaskImage(self, ext, image_item, subdir="masks" + os.sep, alpha=175):
        im_data = self.findImage(subdir=subdir, ext=ext)
        im_RGBA = np.zeros((im_data.shape[0], im_data.shape[1], 4), dtype=np.uint8)
        im_RGBA[:,:,0] = 255
        im_RGBA[:,:,3] = im_data * alpha
        image_item.setImage(im_RGBA)

    def forward(self):
        self.curr_pos += 1
        self.curr_pos = self.curr_pos % len(self.tiflist[self.keylist[self.curr_key]])
        self.updateImages(autoScaleRange=False)
        # self.setWindowTitle(self.tiflist[self.keylist[self.curr_key]][self.curr_pos])
        self.image_name.setText(
            self.tiflist[self.keylist[self.curr_key]][self.curr_pos]
        )

    def backward(self):
        self.curr_pos -= 1
        self.curr_pos = self.curr_pos % len(self.tiflist[self.keylist[self.curr_key]])
        self.updateImages(autoScaleRange=False)
        # self.setWindowTitle(self.tiflist[self.keylist[self.curr_key]][self.curr_pos])
        self.image_name.setText(
            self.tiflist[self.keylist[self.curr_key]][self.curr_pos]
        )

    def prevkey(self):
        self.curr_pos = 0
        self.curr_key -= 1
        self.curr_key = self.curr_key % len(self.keylist)
        self.updateImages(resetHistoRange=True)
        # self.setWindowTitle(self.tiflist[self.keylist[self.curr_key]][self.curr_pos])
        self.image_name.setText(
            self.tiflist[self.keylist[self.curr_key]][self.curr_pos]
        )

    def nextkey(self):
        self.curr_pos = 0
        self.curr_key += 1
        self.curr_key = self.curr_key % len(self.keylist)
        self.updateImages(resetHistoRange=True)
        # self.setWindowTitle(self.tiflist[self.keylist[self.curr_key]][self.curr_pos])
        self.image_name.setText(
            self.tiflist[self.keylist[self.curr_key]][self.curr_pos]
        )

    def update_other_histo_ranges(self, histo):
        levels = histo.getLevels()
        if self.image_2.link_to_first_image_checkbox.isChecked():
            self.image_2.imview.getHistogramWidget().setLevels(
                min=levels[0], max=levels[1]
            )
        if self.image_3.link_to_first_image_checkbox.isChecked():
            self.image_3.imview.getHistogramWidget().setLevels(
                min=levels[0], max=levels[1]
            )
        if self.image_4.link_to_first_image_checkbox.isChecked():
            self.image_4.imview.getHistogramWidget().setLevels(
                min=levels[0], max=levels[1]
            )

    def zlink_2_to_1(self, state):
        if state:
            self.image_2.imview.getHistogramWidget().vb.setYLink(
                self.image_1.imview.getHistogramWidget().vb
            )
        else:
            self.image_2.imview.getHistogramWidget().vb.setYLink(None)

    def zlink_3_to_1(self, state):
        if state:
            self.image_3.imview.getHistogramWidget().vb.setYLink(
                self.image_1.imview.getHistogramWidget().vb
            )
        else:
            self.image_3.imview.getHistogramWidget().vb.setYLink(None)

    def zlink_4_to_1(self, state):
        if state:
            self.image_4.imview.getHistogramWidget().vb.setYLink(
                self.image_1.imview.getHistogramWidget().vb
            )
        else:
            self.image_4.imview.getHistogramWidget().vb.setYLink(None)

    def image_directory_changed(self, directory):
        self.image_directory = directory
        self.update_tiflist(reset=True)

    def output_directory_changed(self, directory):
        self.output_directory = directory
        # self.curr_key = 0
        # self.curr_pos = 0
        self.updateImages()

    def config_button_pressed(self):
        if self.hide_config_button.text() == "Hide Config":
            self.image_1.hide_config()
            self.image_2.hide_config()
            self.image_3.hide_config()
            self.image_4.hide_config()
            self.hide_config_button.setText("Show Config")
        elif self.hide_config_button.text() == "Show Config":
            self.image_1.show_config()
            self.image_2.show_config()
            self.image_3.show_config()
            self.image_4.show_config()
            self.hide_config_button.setText("Hide Config")


def main_GUI():
    app = pg.mkQApp()
    win = KeyPressWindow()
    win.resize(1000, 800)
    win.show()
    app.exec()


if __name__ == "__main__":
    main_GUI()
