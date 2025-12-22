"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the main image view widget for the results UI.
"""


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import os
import glob
import tifffile as tf

from mainUI.UI_settings import Settings


class image_mask:
    """
    Holds the opacity, mask data, color, and 3d RGBA array
    created using that information
    """
    def __init__(self, shape: tuple, color: str):
        self.opacity = 1
        self.__shape = shape
        self.mask_data = np.zeros(shape, dtype=bool)
        self.full_data = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self.color = color
        self._update_color()

    def _update_color(self):
        colorRGB = pg.mkColor(self.color).getRgb()
        self.full_data[:, :, 0] = int(colorRGB[0])
        self.full_data[:, :, 1] = int(colorRGB[1])
        self.full_data[:, :, 2] = int(colorRGB[2])

    def set_color(self, color):
        self.color = color
        self._update_color()

    def set_shape(self, shape):
        self.__shape = shape
        self.mask_data = np.zeros(shape, dtype=bool)
        self.full_data = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        self._update_color()

    def set_data(self, data):
        if data.shape != self.__shape:
            self.set_shape(data.shape)
        self.mask_data = data
        self.full_data[:, :, 3] = data * int(255 * self.opacity)

    def set_opacity(self, opacity):
        self.opacity = opacity
        self.full_data[:, :, 3] = self.mask_data * int(255 * self.opacity)


class MainImageView(pg.GraphicsLayoutWidget):
    """
    Widget showing the current image, its overlay masks, and a set of widgets
    for toggling the visibility and opacity of the masks. These checkboxes and
    spinboxes are not displayed by MainImageView but are contained here due to
    their effects being local to this canvas. They are displayed in the layout
    of the main window instead.
    """
    def __init__(self, settings: Settings, parent=None):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__(parent)
        self.settings = settings
        self.maps_loaded = False
        self.view = self.addPlot(title="")
        self.view.setAspectLocked(True)
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)
        # images_exist = False
        # while not images_exist:
        #     images = glob.glob(self.directory+"/*.tif.metadata")
        #     if len(images) > 0:
        #         del images
        #         images_exist = True
        # del images_exist
        # #print(tiflist)
        # self.image_data = tf.imread(directory + "\\" + tiflist[keylist[curr_key]][curr_pos] + ".tif")
        # self.image = pg.ImageItem(self.image_data)
        self.image_data = np.zeros(self.settings.image_size)
        self.image = pg.ImageItem()

        self.view.addItem(self.image)
        self.intensityBar = pg.HistogramLUTItem()
        self.intensityBar.setImageItem(self.image)
        self.intensityBar.gradient.setColorMap(self.cmap)
        self.intensityBar.gradient.showTicks(show=False)
        self.addItem(self.intensityBar)
        # self.predef_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.predef_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.predef_mask_RGBA,"hotpink")
        self.predef_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["predef_mask"].color
        )
        self.nonpositive_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["nonpositive_mask"].color
        )
        self.outlier_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["outlier_mask"].color
        )  # green
        self.outlier_mask_only_data = image_mask(
            self.settings.image_size, self.settings.colors["outlier_mask"].color
        )
        self.predef_mask = pg.ImageItem(self.predef_mask_data.full_data, levels=None)
        # self.nonzero_mask = pg.ImageItem(self.nonzero_mask_RGBA,levels=None)
        self.outlier_mask = pg.ImageItem(self.outlier_mask_data.full_data, levels=None)
        self.spot_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["spot_mask"].color
        )  # darkcyan
        self.arcs_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["arcs_mask"].color
        )  # maroon
        self.spot_mask = pg.ImageItem(self.spot_mask_data.full_data, levels=None)
        self.arcs_mask = pg.ImageItem(self.arcs_mask_data.full_data, levels=None)

        self.masks = {
            self.predef_mask: [self.predef_mask_data, "_base.tif"],
            # self.nonzero_mask: [self.nonzero_mask_RGBA,self.nonzero_mask_vals,"_om.tif"],
            self.outlier_mask: [self.outlier_mask_data, "_outliermask.tif"],
            self.spot_mask: [self.spot_mask_data, "_spots.tif"],
            self.arcs_mask: [self.arcs_mask_data, "_arcs.tif"],
        }

        self.masks_label = QtWidgets.QLabel("Masks:")
        self.mask_opacity_label = QtWidgets.QLabel("Mask Opacity:")
        self.predef_mask_opacity_box = QtWidgets.QSpinBox()
        self.outlier_mask_opacity_box = QtWidgets.QSpinBox()
        self.spot_mask_opacity_box = QtWidgets.QSpinBox()
        self.arcs_mask_opacity_box = QtWidgets.QSpinBox()
        for mask_opacity_box in [
                self.predef_mask_opacity_box,
                self.outlier_mask_opacity_box,
                self.spot_mask_opacity_box,
                self.arcs_mask_opacity_box,
                ]:
            mask_opacity_box.setMinimum(0)
            mask_opacity_box.setMaximum(100)
            mask_opacity_box.setSingleStep(10)
            mask_opacity_box.setValue(100)
        self.predef_mask_opacity_box.valueChanged.connect(self.predef_mask_opacity_changed)
        self.outlier_mask_opacity_box.valueChanged.connect(self.outlier_mask_opacity_changed)
        self.spot_mask_opacity_box.valueChanged.connect(self.spot_mask_opacity_changed)
        self.arcs_mask_opacity_box.valueChanged.connect(self.arcs_mask_opacity_changed)

        self.view.addItem(self.outlier_mask)
        self.view.addItem(self.spot_mask)
        self.view.addItem(self.arcs_mask)
        self.view.addItem(self.predef_mask)

        self.predef_mask_box = QtWidgets.QCheckBox("Predefined Mask")
        self.mask_box = QtWidgets.QCheckBox("Outlier Mask")
        self.outlier_only_box = QtWidgets.QCheckBox("Only unclassified clusters")
        self.predef_mask_box.setChecked(True)
        self.mask_box.setChecked(True)
        self.predef_mask_box.stateChanged.connect(self.predef_box_changed)
        self.mask_box.stateChanged.connect(self.mask_box_changed)
        self.outlier_only_box.stateChanged.connect(self.outlier_only_changed)
        self.spot_mask_box = QtWidgets.QCheckBox("Spot Mask")
        self.arcs_mask_box = QtWidgets.QCheckBox("Texture Mask")
        self.spot_mask_box.setChecked(True)
        self.arcs_mask_box.setChecked(True)
        self.spot_mask_box.stateChanged.connect(self.spot_box_changed)
        self.arcs_mask_box.stateChanged.connect(self.arcs_box_changed)

        self.tth_map = np.zeros(self.settings.image_size)
        self.azim_map = np.zeros(self.settings.image_size)

        self.tth_circle_data = image_mask(
            self.settings.image_size, self.settings.colors["tth_circle_mask"].color
        )
        self.tth_circle_data.set_opacity(0.5)
        self.tth_circle = pg.ImageItem(self.tth_circle_data.full_data, levels=None)
        self.view.addItem(self.tth_circle)

    def load_maps(self):
        tth_maps = glob.glob(os.path.join(self.settings.output_directory,"maps")+os.sep+"*_2thetamap.tif")
        azim_maps = glob.glob(os.path.join(self.settings.output_directory,"maps")+os.sep+"*_azmmap.tif")
        if len(tth_maps) > 0 and len(azim_maps) > 0:
            self.tth_map = tf.imread(tth_maps[0])
            self.azim_map = tf.imread(azim_maps[0])
            self.maps_loaded = True

    def update_dir(self):
        # Levels: z min and max
        # Range: x, y min and max
        # HistogramRange: visible axis range for z
        self.maps_loaded = False
        self.load_maps()
        self.predef_mask_data.set_color(self.settings.colors["predef_mask"].color)
        self.nonpositive_mask_data.set_color(
            self.settings.colors["nonpositive_mask"].color
        )
        self.outlier_mask_data.set_color(self.settings.colors["outlier_mask"].color)
        self.outlier_mask_only_data.set_color(self.settings.colors["outlier_mask"].color)
        self.arcs_mask_data.set_color(self.settings.colors["arcs_mask"].color)
        self.spot_mask_data.set_color(self.settings.colors["spot_mask"].color)
        if self.maps_loaded:
            self.tth_circle_data.set_color(self.settings.colors["tth_circle_mask"].color)
            self.tth_circle_data.set_shape(self.settings.image_size)
            self.tth_circle.updateImage(self.tth_circle_data.full_data)
        self.update_image_data(xy_reset=True, z_reset=True)
        self.update_masks_data()

    def update_image_data(self, xy_reset=False, z_reset=False):
        # check for flatfield-corrected images first
        if os.path.exists(
            os.path.join(
                self.settings.output_directory,
                "flatfield",
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_flatfield_correct.tif"
            )
        ):
            self.image_data = tf.imread(
                os.path.join(
                    self.settings.output_directory,
                    "flatfield",
                    self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + "_flatfield_correct.tif"
                )
            )
        else:
            self.image_data = tf.imread(
                os.path.join(
                    self.settings.image_directory,
                    self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + ".tif"
                )
            )
        if z_reset:
            maxval = np.percentile(self.image_data, 99.9)
            self.image.updateImage(
                self.image_data, autoRange=xy_reset, autoLevels=False
            )
            self.intensityBar.setLevels(min=0.0, max=maxval)
        else:
            self.image.updateImage(
                self.image_data,
                autoRange=xy_reset,
                autoLevels=z_reset,
                autoHistogramRange=z_reset,
            )

    def update_masks_data(self):
        for mask,vals in self.masks.items():
            file_name = os.path.join(
                self.settings.output_directory,
                "masks",
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + vals[1]
            )
            # Handle cases where the file exists but is still being written or is otherwise corrupted
            try:
                # vals[1] = tf.imread(file_name)
                vals[0].set_data(tf.imread(file_name))
            except:
                vals[0].set_shape(self.settings.image_size)
            mask.updateImage(vals[0].full_data)
        self.outlier_mask_only_data.set_data(self.outlier_mask_data.mask_data ^ self.arcs_mask_data.mask_data ^ self.spot_mask_data.mask_data)
        if self.outlier_only_box.isChecked():
            self.outlier_mask.updateImage(self.outlier_mask_only_data.full_data)

    def update_tth_circle(self, tth, width=0.03):
        circle = (self.tth_map > tth - width) & (self.tth_map < tth + width)
        self.tth_circle_data.set_data(circle)
        self.tth_circle.updateImage(self.tth_circle_data.full_data)

    def mask_box_changed(self):
        if self.mask_box.isChecked():
            # self.view.addItem(self.outlier_mask)
            self.outlier_mask.setVisible(True)
        else:
            # self.view.removeItem(self.outlier_mask)
            self.outlier_mask.setVisible(False)

    def outlier_only_changed(self):
        if self.outlier_only_box.isChecked():
            self.outlier_mask.updateImage(self.outlier_mask_only_data.full_data)
        else:
            self.outlier_mask.updateImage(self.outlier_mask_data.full_data)

    def predef_box_changed(self):
        if self.predef_mask_box.isChecked():
            # self.view.addItem(self.predef_mask)
            self.predef_mask.setVisible(True)
        else:
            # self.view.removeItem(self.predef_mask)
            self.predef_mask.setVisible(False)

    def spot_box_changed(self):
        if self.spot_mask_box.isChecked():
            # self.view.addItem(self.spot_mask)
            self.spot_mask.setVisible(True)
        else:
            # self.view.removeItem(self.spot_mask)
            self.spot_mask.setVisible(False)

    def arcs_box_changed(self):
        if self.arcs_mask_box.isChecked():
            # self.view.addItem(self.arcs_mask)
            self.arcs_mask.setVisible(True)
        else:
            # self.view.removeItem(self.arcs_mask)
            self.arcs_mask.setVisible(False)

    def predef_mask_opacity_changed(self, evt):
        self.predef_mask_data.set_opacity(evt / 100)
        self.predef_mask.updateImage(self.predef_mask_data.full_data)
    
    def outlier_mask_opacity_changed(self, evt):
        self.outlier_mask_data.set_opacity(evt / 100)
        self.outlier_mask_only_data.set_opacity(evt / 100)
        if self.outlier_only_box.isChecked():
            self.outlier_mask.updateImage(self.outlier_mask_only_data.full_data)
        else:
            self.outlier_mask.updateImage(self.outlier_mask_data.full_data)

    def spot_mask_opacity_changed(self, evt):
        self.spot_mask_data.set_opacity(evt / 100)
        self.spot_mask.updateImage(self.spot_mask_data.full_data)

    def arcs_mask_opacity_changed(self, evt):
        self.arcs_mask_data.set_opacity(evt / 100)
        self.arcs_mask.updateImage(self.arcs_mask_data.full_data)

