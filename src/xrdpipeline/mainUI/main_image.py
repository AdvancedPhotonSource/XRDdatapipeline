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
    def __init__(self, settings: Settings, parent=None):
        # global tiflist, keylist, curr_key, curr_pos
        super().__init__(parent)
        self.settings = settings
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
        # Call update mask data instead of reading these in
        # self.predef_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_predef.tif")
        # self.nonzero_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.nonzero_mask_vals = np.zeros(self.image_data.shape)
        # self.outlier_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.outlier_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.outlier_mask_RGBA,"green")
        self.outlier_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["outlier_mask"].color
        )  # green
        # self.nonzero_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_maskpolnonzero.tif")
        # TODO: nonzero version
        # self.nonzero_mask_RGBA[:,:,3] = tf.imread("masks\\" + tiflist[keylist[curr_key]][curr_pos] + "_om.tif")
        self.predef_mask = pg.ImageItem(self.predef_mask_data.full_data, levels=None)
        # self.nonzero_mask = pg.ImageItem(self.nonzero_mask_RGBA,levels=None)
        self.outlier_mask = pg.ImageItem(self.outlier_mask_data.full_data, levels=None)
        # self.spot_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.arcs_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.spot_mask_vals = np.zeros(self.image_data.shape)
        # self.arcs_mask_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.spot_mask_RGBA,"darkcyan")
        # self.update_mask_color(self.arcs_mask_RGBA,"maroon")
        self.spot_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["spot_mask"].color
        )  # darkcyan
        self.arcs_mask_data = image_mask(
            self.settings.image_size, self.settings.colors["arcs_mask"].color
        )  # maroon
        self.spot_mask = pg.ImageItem(self.spot_mask_data.full_data, levels=None)
        self.arcs_mask = pg.ImageItem(self.arcs_mask_data.full_data, levels=None)
        # #closed masks
        # self.closed_mask_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_spots_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_arcs_RGBA = np.zeros((self.image_data.shape[0],self.image_data.shape[1],4),dtype=np.uint8)
        # self.closed_mask_vals = np.zeros(self.image_data.shape)
        # self.closed_spots_vals = np.zeros(self.image_data.shape)
        # self.closed_arcs_vals = np.zeros(self.image_data.shape)
        # self.update_mask_color(self.closed_mask_RGBA,"lime") #red
        # self.update_mask_color(self.closed_spots_RGBA,"cyan") #cyan
        # self.update_mask_color(self.closed_arcs_RGBA,"red") #pink
        # self.closed_mask = pg.ImageItem(self.closed_mask_RGBA,levels=None)
        # self.closed_spots = pg.ImageItem(self.closed_spots_RGBA,levels=None)
        # self.closed_arcs = pg.ImageItem(self.closed_arcs_RGBA,levels=None)

        self.masks = {
            self.predef_mask: [self.predef_mask_data, "_base.tif"],
            # self.nonzero_mask: [self.nonzero_mask_RGBA,self.nonzero_mask_vals,"_om.tif"],
            self.outlier_mask: [self.outlier_mask_data, "_outliermask.tif"],
            self.spot_mask: [self.spot_mask_data, "_spots.tif"],
            self.arcs_mask: [self.arcs_mask_data, "_arcs.tif"],
            # self.closed_mask: [self.closed_mask_RGBA,self.closed_mask_vals,"_closedmask.tif"],
            # self.closed_spots: [self.closed_spots_RGBA,self.closed_spots_vals,"_spotsclosed.tif"],
            # self.closed_arcs: [self.closed_arcs_RGBA,self.closed_arcs_vals,"_arcsclosed.tif"],
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

        # self.update_masks_data()

        # Initial draw places non-closed above closed so all visible
        # self.view.addItem(self.closed_mask)
        self.view.addItem(self.outlier_mask)
        # self.view.addItem(self.closed_spots)
        self.view.addItem(self.spot_mask)
        # self.view.addItem(self.closed_arcs)
        self.view.addItem(self.arcs_mask)
        self.view.addItem(self.predef_mask)

        self.predef_mask_box = QtWidgets.QCheckBox("Predefined Mask")
        self.mask_box = QtWidgets.QCheckBox("Outlier Mask")
        self.predef_mask_box.setChecked(True)
        self.mask_box.setChecked(True)
        self.predef_mask_box.stateChanged.connect(self.predef_box_changed)
        self.mask_box.stateChanged.connect(self.mask_box_changed)
        self.spot_mask_box = QtWidgets.QCheckBox("Spot Mask")
        self.arcs_mask_box = QtWidgets.QCheckBox("Texture Mask")
        self.spot_mask_box.setChecked(True)
        self.arcs_mask_box.setChecked(True)
        self.spot_mask_box.stateChanged.connect(self.spot_box_changed)
        self.arcs_mask_box.stateChanged.connect(self.arcs_box_changed)
        # self.closedmask_box = QtWidgets.QCheckBox("Closed Mask")
        # self.closedspots_box = QtWidgets.QCheckBox("Closed Spots Mask")
        # self.closedarcs_box = QtWidgets.QCheckBox("Closed Arcs Mask")
        # self.closedmask_box.setChecked(True)
        # self.closedspots_box.setChecked(True)
        # self.closedarcs_box.setChecked(True)
        # self.closedmask_box.stateChanged.connect(self.closedmask_box_changed)
        # self.closedspots_box.stateChanged.connect(self.closedspots_box_changed)
        # self.closedarcs_box.stateChanged.connect(self.closedarcs_box_changed)

        # one of the pyqtgraph examples shows the setDrawKernel() option for an image. Could make a zero mask image that gets modified by the draw kernel (brush; probably just want [[1.0]] but can resize) for a mask.

        # TODO: There might be more than one file here if they tried a previous experiment. Prompt for imctrl file name.
        # Wait until these files exist.
        # maps_loaded = False
        # while not maps_loaded:
        #     maps = glob.glob(self.settings.directory + "\\maps\\*.tif")
        #     if len(maps) > 0:
        #         time.sleep(1)
        #         del maps
        #         maps_loaded = True
        # del maps_loaded
        # self.tth_map = tf.imread(glob.glob(self.settings.directory+"\\maps\\*_2thetamap.tif")[0])
        # self.azim_map = tf.imread(glob.glob(self.settings.directory+"\\maps\\*_azmmap.tif")[0])
        self.tth_map = np.zeros(self.settings.image_size)
        self.azim_map = np.zeros(self.settings.image_size)

        # one option is to create a QGraphicsEllipseItem(x,y,width,height), but it would be better to use the 2th map
        # self.tth_circle_RGBA = np.zeros_like(self.spot_mask_RGBA)
        # self.update_mask_color(self.tth_circle_RGBA,"white")
        self.tth_circle_data = image_mask(
            self.settings.image_size, self.settings.colors["tth_circle_mask"].color
        )
        self.tth_circle_data.set_opacity(0.5)
        self.tth_circle = pg.ImageItem(self.tth_circle_data.full_data, levels=None)
        self.view.addItem(self.tth_circle)

    def update_dir(self):
        # Levels: z min and max
        # Range: x, y min and max
        # HistogramRange: visible axis range for z
        self.tth_map = tf.imread(
            glob.glob(os.path.join(self.settings.output_directory,"maps")+os.sep+"*_2thetamap.tif")[0]
        )
        self.azim_map = tf.imread(
            glob.glob(os.path.join(self.settings.output_directory,"maps")+os.sep+"*_azmmap.tif")[0]
        )
        self.predef_mask_data.set_color(self.settings.colors["predef_mask"].color)
        self.nonpositive_mask_data.set_color(
            self.settings.colors["nonpositive_mask"].color
        )
        self.outlier_mask_data.set_color(self.settings.colors["outlier_mask"].color)
        self.arcs_mask_data.set_color(self.settings.colors["arcs_mask"].color)
        self.spot_mask_data.set_color(self.settings.colors["spot_mask"].color)
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

    # def update_mask_color(self,mask,color="red"):
    #     colorRGB = pg.mkColor(color).getRgb()
    #     mask[:,:,0] = int(colorRGB[0])
    #     mask[:,:,1] = int(colorRGB[1])
    #     mask[:,:,2] = int(colorRGB[2])

    def update_masks_data(self):
        # global tiflist, keylist, curr_key, curr_pos
        for mask,vals in self.masks.items():
            #print(tiflist[keylist[curr_key]][curr_pos])
            file_name = os.path.join(
                self.settings.output_directory,
                "masks",
                self.settings.tiflist[self.settings.keylist[self.settings.curr_key]][self.settings.curr_pos] + vals[1]
            )
            #if os.path.exists(file_name):
            #    vals[0][:,:,3] = tf.imread(file_name)
            # else:
            #    vals[0][:,:,3] = 0
            # Handle cases where the file exists but is still being written or is otherwise corrupted
            try:
                # vals[1] = tf.imread(file_name)
                vals[0].set_data(tf.imread(file_name))
            except:
                vals[0].set_shape(self.settings.image_size)
            # vals[0][:,:,3] = vals[1]*int(255*self.mask_opacity_box.value()/100)
            mask.updateImage(vals[0].full_data)

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

    # def closedmask_box_changed(self):
    #     if self.closedmask_box.isChecked():
    #         #self.view.addItem(self.closed_mask)
    #         self.closed_mask.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_mask)
    #         self.closed_mask.setVisible(False)
    # def closedspots_box_changed(self):
    #     if self.closedspots_box.isChecked():
    #         #self.view.addItem(self.closed_spots)
    #         self.closed_spots.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_spots)
    #         self.closed_spots.setVisible(False)
    # def closedarcs_box_changed(self):
    #     if self.closedarcs_box.isChecked():
    #         #self.view.addItem(self.closed_arcs)
    #         self.closed_arcs.setVisible(True)
    #     else:
    #         #self.view.removeItem(self.closed_arcs)
    #         self.closed_arcs.setVisible(False)
 
    def predef_mask_opacity_changed(self, evt):
        self.predef_mask_data.set_opacity(evt / 100)
        self.predef_mask.updateImage(self.predef_mask_data.full_data)
    
    def outlier_mask_opacity_changed(self, evt):
        self.outlier_mask_data.set_opacity(evt / 100)
        self.outlier_mask.updateImage(self.outlier_mask_data.full_data)

    def spot_mask_opacity_changed(self, evt):
        self.spot_mask_data.set_opacity(evt / 100)
        self.spot_mask.updateImage(self.spot_mask_data.full_data)

    def arcs_mask_opacity_changed(self, evt):
        self.arcs_mask_data.set_opacity(evt / 100)
        self.arcs_mask.updateImage(self.arcs_mask_data.full_data)

