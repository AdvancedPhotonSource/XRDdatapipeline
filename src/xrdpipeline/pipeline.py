import argparse
import copy
import glob
import os
import re
import sys
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

import PySide6
import skimage as ski
import tifffile as tf

import torch
from PIL import Image
from pyqtgraph.Qt import QtCore, QtWidgets

from scipy import spatial
from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
bindist_dir = os.path.join(script_dir, "bindist")
# Add 'bindist' to the beginning of sys.path
print(bindist_dir)
sys.path.insert(0, bindist_dir)
# import polymask as pm


# trig functions using degrees
# numpy versions
def npsind(x):
    return np.sin(x * np.pi / 180.0)


def npasind(x):
    return 180.0 * np.arcsin(x) / np.pi


def npcosd(x):
    return np.cos(x * np.pi / 180.0)


def npacosd(x):
    return 180.0 * np.arccos(x) / np.pi


def nptand(x):
    return np.tan(x * np.pi / 180.0)


def npatand(x):
    return 180.0 * np.arctan(x) / np.pi


def npatan2d(y, x):
    return 180.0 * np.arctan2(y, x) / np.pi


# recreating xye export function
# TODO: use numpy or the like to write this faster
def Export_xye(name, data, location, error=True):
    with open(location, "w") as outfile:
        outfile.write("/*\n")
        outfile.write("# {0}\n".format(name))
        outfile.write("*/\n")
        # data = integral['data'][1]
        for i in range(len(data[0])):
            if error:
                outfile.write(
                    "{x}\t{y}\t{e}\n".format(x=data[0][i], y=data[1][i], e=data[2][i])
                )
            else:
                outfile.write("{x}\t{y}\t{e}\n".format(x=data[0][i], y=data[1][i], e=0))


# load data from file
def load_image_tifffile(imloc):
    image = tf.imread(imloc)
    return image


def load_image(imloc):
    image = ski.io.imread(imloc)
    return image


class image_monitor(RegexMatchingEventHandler):
    def __init__(self, queue):
        # dir\name_number_ext.tif or dir\name-number_ext.tif
        #'number' may be 00000 or xxxxx_xxxxx or xxxxx-xxxxx
        #'_ext' not on base images
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif.metadata$"
        # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif$"
        reg_image = r"(?P<directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
        # reg for integral data files

        ###
        regs = [reg_image]
        RegexMatchingEventHandler.__init__(self, regexes=regs)
        self.queue = queue

    def on_created(self, event):
        print("New file at {0}".format(event.src_path))
        results = [r.match(event.src_path) for r in self.regexes]
        # print(results[0].group(0,1,2,3,4))
        print(
            "Directory: {0}, Name: {1}, Number: {2}, Extension: {3}".format(
                results[0].group("directory"),
                results[0].group("name"),
                results[0].group("number"),
                results[0].group("ext"),
            )
        )
        # number -> actual int
        # number_int = results[0].group("number").remove("-").remove("_")
        # number_int = int(number_int)

        # Add file path to queue, stripping ".metadata"
        # TODO: WARNING: REMOVE THE 10X FOR ACTUAL RUNS
        # Done for testing 10k input without requiring 10x disk space
        # for i in range(10000):
        #    self.queue.append([event.src_path[:-9],results[0].group("name"),results[0].group("number")])
        # self.queue.append([event.src_path[:-9],results[0].group("name"),results[0].group("number")])
        self.queue.append(
            [
                event.src_path,
                results[0].group("name"),
                results[0].group("number"),
                results[0].group("ext"),
            ]
        )
        # self.queue.put([event.src_path,results.group("name"),results.group("number")])


class file_select(QtWidgets.QWidget):
    def __init__(self, label, default_text=None, isdir=False, startdir=".", ext=None):
        super().__init__()
        self.setMinimumWidth(600)
        # self.label = QtWidgets.QLabel(label)
        self.file_select_button = QtWidgets.QPushButton(label)
        # self.file_select_button = QtWidgets.QPushButton(self.default_text)
        self.file_name = QtWidgets.QLabel(default_text)
        self.isdir = isdir
        self.startdir = startdir
        self.ext = ext

        self.layout = QtWidgets.QGridLayout()
        # self.layout.addWidget(self.label,0,0)
        # self.layout.addWidget(self.file_select_button,0,1,1,3)
        self.layout.addWidget(self.file_select_button, 0, 0)
        self.layout.addWidget(self.file_name, 0, 1, 1, 3)
        self.setLayout(self.layout)

        self.file_select_button.released.connect(self.select_file)

    def select_file(self):
        if self.isdir:
            location = QtWidgets.QFileDialog.getExistingDirectory(
                None, "Select Directory"
            )
            # self.file_select_button.setText(location)
            self.file_name.setText(location)
        else:
            location = QtWidgets.QFileDialog.getOpenFileName(
                None, "Select File", self.startdir, self.ext
            )
            # self.file_select_button.setText(location[0])
            self.file_name.setText(location[0])


# G2sc; creates cache
def read_image(imagename):
    # image_data = tf.imread(imagename)
    image_data = load_image(imagename)
    Data = {}
    Data["type"] = "PWDR"
    # Data['color'] = GSASIIpath.GetConfigValue('Contour_color','Paired')
    if "tilt" not in Data:  # defaults if not preset in e.g. Bruker importer
        Data["tilt"] = 0.0
        Data["rotation"] = 0.0
        Data["pixLimit"] = 20
        Data["calibdmin"] = 0.5
        Data["cutoff"] = 10.0
    Data["showLines"] = False
    Data["calibskip"] = 0
    Data["ring"] = []
    Data["rings"] = []
    Data["edgemin"] = 100000000
    Data["ellipses"] = []
    Data["GonioAngles"] = [0.0, 0.0, 0.0]
    Data["DetDepth"] = 0.0
    Data["DetDepthRef"] = False
    Data["calibrant"] = ""
    Data["IOtth"] = [5.0, 50.0]
    Data["LRazimuth"] = [0.0, 180.0]
    Data["azmthOff"] = 0.0
    Data["outChannels"] = 2500
    Data["outAzimuths"] = 1
    Data["centerAzm"] = False
    Data["fullIntegrate"] = True
    Data["setRings"] = False
    Data["background image"] = ["", -1.0]
    Data["dark image"] = ["", -1.0]
    Data["Flat Bkg"] = 0.0
    Data["Oblique"] = [0.5, False]
    Data["varyList"] = {
        "dist": True,
        "det-X": True,
        "det-Y": True,
        "tilt": True,
        "phi": True,
        "dep": False,
        "wave": False,
    }
    Data["setDefault"] = False
    # Imax is image intensity maximum
    Imax = np.max(image_data)
    Data["range"] = [(0, Imax), [0, Imax]]
    # size, samplechangerpos, det2theta, ImageTag, formatName
    Data["size"] = [image_data.shape[0], image_data.shape[1]]
    print(Data["size"])
    Data["samplechangerpos"] = None
    Data["det2theta"] = 0.0
    ImgDict = {}
    ImgDict["Comments"] = {}
    with tf.TiffFile(imagename) as tif:
        for tag in tif.pages[0].tags:
            ImgDict["Comments"][tag.name] = tag.value
    ImgDict["Image Controls"] = Data
    ImgDict["Masks"] = {
        "Points": [],
        "Rings": [],
        "Arcs": [],
        "Polygons": [],
        "Frames": [],
        "Thresholds": [(0, Imax), [0, Imax]],
        "SpotMask": {"esdMul": 3.0, "spotMask": None},
    }
    ImgDict["Stress/Strain"] = {
        "Type": "True",
        "d-zero": [],
        "Sample phi": 0.0,
        "Sample z": 0.0,
        "Sample load": 0.0,
    }
    # ImgDict['image'] = image_data
    ImgDict["image"] = image_data
    ImgDict["corrected_image"] = None

    return ImgDict


# G2fil; updates cache from imctrl file
def LoadControls(Slines, data):
    "Read values from a .imctrl (Image Controls) file"
    cntlList = [
        "color",
        "wavelength",
        "distance",
        "tilt",
        "invert_x",
        "invert_y",
        "type",
        "Oblique",
        "fullIntegrate",
        "outChannels",
        "outAzimuths",
        "LRazimuth",
        "IOtth",
        "azmthOff",
        "DetDepth",
        "calibskip",
        "pixLimit",
        "cutoff",
        "calibdmin",
        "Flat Bkg",
        "varyList",
        "setdist",
        "PolaVal",
        "SampleAbs",
        "dark image",
        "background image",
        "twoth",
    ]
    save = {}
    for S in Slines:
        if S[0] == "#":
            continue
        [key, val] = S.strip().split(":", 1)
        if key in [
            "type",
            "calibrant",
            "binType",
            "SampleShape",
            "color",
        ]:  # strings
            save[key] = val
        elif key in [
            "varyList",
        ]:
            save[key] = eval(val)  # dictionary
        elif key in ["rotation"]:
            save[key] = float(val)
        elif key in [
            "center",
        ]:
            if "," in val:
                save[key] = eval(val)
            else:
                vals = val.strip("[] ").split()
                save[key] = [float(vals[0]), float(vals[1])]
        elif key in cntlList:
            save[key] = eval(val)
    data.update(save)


# G2fil; used to read in user-defined GSAS-II masks
def readMasks(filename, masks, ignoreThreshold):
    """Read a GSAS-II masks file"""
    File = open(filename, "r")
    save = {}
    oldThreshold = masks["Thresholds"][0]
    S = File.readline()
    while S:
        if S[0] == "#":
            S = File.readline()
            continue
        [key, val] = S.strip().split(":", 1)
        if key in ["Points", "Rings", "Arcs", "Polygons", "Frames", "Thresholds"]:
            if ignoreThreshold and key == "Thresholds":
                S = File.readline()
                continue
            save[key] = eval(val)
            if key == "Thresholds":
                save[key][0] = oldThreshold
                save[key][1][1] = min(oldThreshold[1], save[key][1][1])
        S = File.readline()
    File.close()
    masks.update(save)
    # CleanupMasks
    for key in ["Points", "Rings", "Arcs", "Polygons"]:
        masks[key] = masks.get(key, [])
        masks[key] = [i for i in masks[key] if len(i)]


# G2img; called by MakeUseMask for integration
def MakeMaskMap(data, masks, iLim, jLim, tamp):
    """Makes a mask array from masking parameters that are not determined by
    image calibration parameters or the image intensities. Thus this uses
    mask Frames, Polygons and Lines settings (but not Thresholds, Rings or
    Arcs). Used on a rectangular section of an image (must be 1024x1024 or
    smaller, as dictated by module polymask) where the size is determined
    by iLim and jLim.

    :param dict data: GSAS-II image data object (describes the image)
    :param list iLim: boundary along x-pixels
    :param list jLim: boundary along y-pixels
    :param np.array tamp: all-zero pixel mask array used in Polymask
    :returns: array[nI,nJ] TA: 2-theta, azimuth, 2 geometric corrections
    """
    pixelSize = data["pixelSize"]
    scalex = pixelSize[0] / 1000.0
    scaley = pixelSize[1] / 1000.0

    tay, tax = np.mgrid[
        iLim[0] + 0.5 : iLim[1] + 0.5, jLim[0] + 0.5 : jLim[1] + 0.5
    ]  # bin centers not corners
    tax = np.asfarray(tax * scalex, dtype=np.float32).flatten()
    tay = np.asfarray(tay * scaley, dtype=np.float32).flatten()
    nI = iLim[1] - iLim[0]
    nJ = jLim[1] - jLim[0]
    # make position masks here
    frame = masks["Frames"]
    tam = np.ma.make_mask_none((nI * nJ))
    if frame:
        tam = np.ma.mask_or(
            tam,
            np.ma.make_mask(
                pm.polymask(nI * nJ, tax, tay, len(frame), frame, tamp)[: nI * nJ]
            )
            ^ True,
        )
    polygons = masks["Polygons"]
    for polygon in polygons:
        if polygon:
            tam = np.ma.mask_or(
                tam,
                np.ma.make_mask(
                    pm.polymask(nI * nJ, tax, tay, len(polygon), polygon, tamp)[
                        : nI * nJ
                    ]
                ),
            )
    for X, Y, rsq in masks["Points"].T:
        tam = np.ma.mask_or(
            tam, np.ma.getmask(np.ma.masked_less((tax - X) ** 2 + (tay - Y) ** 2, rsq))
        )
    if tam.shape:
        tam = np.reshape(tam, (nI, nJ))
    else:
        tam = np.ma.make_mask_none((nI, nJ))
    for xline in masks.get("Xlines", []):  # a y pixel position
        if iLim[0] <= xline <= iLim[1]:
            tam[xline - iLim[0], :] = True
    for yline in masks.get("Ylines", []):  # a x pixel position
        if jLim[0] <= yline <= jLim[1]:
            tam[:, yline - jLim[0]] = True
    return tam  # position mask


# G2img; used in integration
def MakeUseMask(data, masks, blkSize=128):
    """Precomputes a set of blocked mask arrays for the mask elements
    that do not depend on the instrument controls (see :func:`MakeMaskMap`).
    This computation is done optionally, but provides speed as the results
    from this can be cached to avoid recomputation for a series of images
    with the same mask parameters.

    :param np.array data: specifies mask parameters for an image
    :param int blkSize: a blocksize that is selected for speed
    :returns: a list of TA blocks
    """
    Masks = copy.deepcopy(masks)
    Masks["Points"] = np.array(Masks["Points"]).T  # get spots as X,Y,R arrays
    if np.any(masks["Points"]):
        Masks["Points"][2] = np.square(Masks["Points"][2] / 2.0)
    Nx, Ny = data["size"]
    nXBlks = (Nx - 1) // blkSize + 1
    nYBlks = (Ny - 1) // blkSize + 1
    useMask = []
    tamp = np.ma.make_mask_none(
        (1024 * 1024)
    )  # NB: this array size used in the fortran polymask module
    for iBlk in range(nYBlks):
        iBeg = iBlk * blkSize
        iFin = min(iBeg + blkSize, Ny)
        useMaskj = []
        for jBlk in range(nXBlks):
            jBeg = jBlk * blkSize
            jFin = min(jBeg + blkSize, Nx)
            mask = MakeMaskMap(
                data, Masks, (iBeg, iFin), (jBeg, jFin), tamp
            )  # 2-theta & azimuth arrays & create position mask
            useMaskj.append(mask)
        useMask.append(useMaskj)
    return useMask


# G2sc
# Readers = {'Pwdr':[], 'Phase':[], 'Image':[]}
# Readers['Pwdr'] = G2fil.LoadImportRoutines("pwd", "Powder_Data")
# Readers['Phase'] = G2fil.LoadImportRoutines("phase", "Phase")
# Readers['Image'] = G2fil.LoadImportRoutines("img", "Image")


# G2img; used in integration
def MakeUseTA(data, blkSize=128):
    """Precomputes the set of blocked arrays for 2theta-azimuth mapping from
    the controls settings of the current image for image integration.
    This computation is done optionally, but provides speed as the results
    from this can be cached to avoid recomputation for a series of images
    with the same calibration parameters.

    :param np.array data: specifies parameters for an image
    :param int blkSize: a blocksize that is selected for speed
    :returns: a list of TA blocks
    """
    Nx, Ny = data["size"]
    nXBlks = (Nx - 1) // blkSize + 1
    nYBlks = (Ny - 1) // blkSize + 1
    useTA = []
    for iBlk in range(nYBlks):
        iBeg = iBlk * blkSize
        iFin = min(iBeg + blkSize, Ny)
        useTAj = []
        for jBlk in range(nXBlks):
            jBeg = jBlk * blkSize
            jFin = min(jBeg + blkSize, Nx)
            TA = Make2ThetaAzimuthMap(
                data, (iBeg, iFin), (jBeg, jFin)
            )  # 2-theta & azimuth arrays & create position mask
            TA = np.dstack(
                (
                    np.ma.getdata(TA[1]),
                    np.ma.getdata(TA[0]),
                    np.ma.getdata(TA[2]),
                    np.ma.getdata(TA[3]),
                )
            )  # azimuth, 2-theta, dist, pol
            TAr = [
                i.squeeze() for i in np.dsplit(TA, 4)
            ]  # azimuth, 2-theta, dist**2/d0**2, pol
            useTAj.append(TAr)
        useTA.append(useTAj)
    return useTA


# G2img; calcs maps, is used in GeneratePixelMask, is used by MakeUseTA for integration blocks
def Make2ThetaAzimuthMap(data, iLim, jLim):  # most expensive part of integration!
    """Makes a set of matrices that provide the 2-theta, azimuth and geometric
    correction values for each pixel in an image taking into account the
    detector orientation. Can be used for the entire image or a rectangular
    section of an image (determined by iLim and jLim).

    This is used in two ways. For image integration, the computation is done
    over blocks of fixed size (typically 128 or 256 pixels) but for pixel mask
    generation, the two-theta matrix for all pixels is computed. Note that
    for integration, this routine will be called to generate sections as needed
    or may be called by :func:`MakeUseTA`, which creates all sections at
    once, so they can be reused multiple times.

    :param dict data: GSAS-II image data object (describes the image)
    :param list iLim: boundary along x-pixels
    :param list jLim: boundary along y-pixels
    :returns: TA, array[4,nI,nJ]: 2-theta, azimuth, 2 geometric corrections
    """
    pixelSize = data["pixelSize"]
    scalex = pixelSize[0] / 1000.0
    scaley = pixelSize[1] / 1000.0
    tay, tax = np.mgrid[
        iLim[0] + 0.5 : iLim[1] + 0.5, jLim[0] + 0.5 : jLim[1] + 0.5
    ]  # bin centers not corners
    tax = np.asfarray(tax * scalex, dtype=np.float32).flatten()
    tay = np.asfarray(tay * scaley, dtype=np.float32).flatten()
    nI = iLim[1] - iLim[0]
    nJ = jLim[1] - jLim[0]
    TA = np.empty((4, nI, nJ))
    if data["det2theta"]:
        TA[:3] = np.array(
            GetTthAzmG(np.reshape(tax, (nI, nJ)), np.reshape(tay, (nI, nJ)), data)
        )  # includes geom. corr. as dist**2/d0**2 - most expensive step
    else:
        TA[:3] = np.array(
            GetTthAzmG2(np.reshape(tax, (nI, nJ)), np.reshape(tay, (nI, nJ)), data)
        )  # includes geom. corr. as dist**2/d0**2 - most expensive step
    TA[1] = np.where(TA[1] < 0, TA[1] + 360, TA[1])
    TA[3] = Polarization(data["PolaVal"][0], TA[0], TA[1] - 90.0)[0]
    return TA  # 2-theta, azimuth & geom. corr. arrays


# G2pwd; used in Make2ThetaAzimuthMap
def Polarization(Pola, Tth, Azm=0.0):
    """   Calculate angle dependent x-ray polarization correction (not scaled correctly!)

    :param Pola: polarization coefficient e.g 1.0 fully polarized, 0.5 unpolarized
    :param Azm: azimuthal angle e.g. 0.0 in plane of polarization - can be numpy array
    :param Tth: 2-theta scattering angle - can be numpy array
      which (if either) of these is "right"?
    :return: (pola, dpdPola) - both 2-d arrays
      * pola = ((1-Pola)*npcosd(Azm)**2+Pola*npsind(Azm)**2)*npcosd(Tth)**2+ \
        (1-Pola)*npsind(Azm)**2+Pola*npcosd(Azm)**2
      * dpdPola: derivative needed for least squares

    """
    cazm = npcosd(Azm) ** 2
    sazm = npsind(Azm) ** 2
    pola = (
        ((1.0 - Pola) * cazm + Pola * sazm) * npcosd(Tth) ** 2
        + (1.0 - Pola) * sazm
        + Pola * cazm
    )
    dpdPola = -(npsind(Tth) ** 2) * (sazm - cazm)
    return pola, dpdPola


# G2sc; used for GeneratePixelMask
def MaskFrameMask(img):
    """Computes a Frame mask from map input for the current image to be
    used for a pixel mask computation in
    :meth:`~G2Image.GeneratePixelMask`.
    This is optional, as if not supplied, mask computation will compute
    this, but this is a relatively slow computation and the
    results computed here can be reused for other images that have the
    same calibration parameters.
    """
    Controls = img["Image Controls"]
    Masks = img["Masks"]
    frame = Masks["Frames"]
    if img["corrected_image"] is not None:
        ImageZ = img["corrected_image"]
    else:
        ImageZ = _getCorrImage(img)
    tam = np.ma.make_mask_none(ImageZ.shape)
    if frame:
        tam = np.ma.mask_or(tam, MakeFrameMask(Controls, frame))
    return tam


# G2img; used in GeneratePixelMask. Only need this one or next?
def MakeFrameMask(data, frame):
    """Assemble a Frame mask for a image, according to the input supplied.
    Note that this requires use of the Fortran polymask routine that is limited
    to 1024x1024 arrays, so this computation is done in blocks (fixed at 512)
    and the master image is assembled from that.

    :param dict data: Controls for an image. Used to find the image size
      and the pixel dimensions.
    :param list frame: Frame parameters, typically taken from ``Masks['Frames']``.
    :returns: a mask array with dimensions matching the image Controls.
    """
    import polymask as pm

    pixelSize = data["pixelSize"]
    scalex = pixelSize[0] / 1000.0
    scaley = pixelSize[1] / 1000.0
    blkSize = 512
    Nx, Ny = data["size"]
    nXBlks = (Nx - 1) // blkSize + 1
    nYBlks = (Ny - 1) // blkSize + 1
    tam = np.ma.make_mask_none(data["size"])
    for iBlk in range(nXBlks):
        iBeg = iBlk * blkSize
        iFin = min(iBeg + blkSize, Nx)
        for jBlk in range(nYBlks):
            jBeg = jBlk * blkSize
            jFin = min(jBeg + blkSize, Ny)
            nI = iFin - iBeg
            nJ = jFin - jBeg
            tax, tay = np.mgrid[
                iBeg + 0.5 : iFin + 0.5, jBeg + 0.5 : jFin + 0.5
            ]  # bin centers not corners
            tax = np.asfarray(tax * scalex, dtype=np.float32)
            tay = np.asfarray(tay * scaley, dtype=np.float32)
            tamp = np.ma.make_mask_none((1024 * 1024))
            tamp = (
                np.ma.make_mask(
                    pm.polymask(
                        nI * nJ, tax.flatten(), tay.flatten(), len(frame), frame, tamp
                    )[: nI * nJ]
                )
                ^ True
            )  # switch to exclude around frame
            if tamp.shape:
                tamp = np.reshape(tamp[: nI * nJ], (nI, nJ))
                tam[iBeg:iFin, jBeg:jFin] = np.ma.mask_or(
                    tamp[0:nI, 0:nJ], tam[iBeg:iFin, jBeg:jFin]
                )
            else:
                tam[iBeg:iFin, jBeg:jFin] = True
    return tam.T


# G2sc; used in GeneratePixelMask, MaskFrameMask, and integration
def _getCorrImage(img):
    """Gets image & applies dark, background & flat background corrections.
    based on :func:`GSASIIimgGUI.GetImageZ`. Expected to be for internal
    use only.

    :param list ImageReaderlist: list of Reader objects for images
    :param object proj: references a :class:`G2Project` project
    :param imageRef: A reference to the desired image in the project.
      Either the Image tree name (str), the image's index (int) or
      a image object (:class:`G2Image`)

    :return: array sumImg: corrected image for background/dark/flat back
    """
    ImgObj = img
    Controls = img["Image Controls"]
    formatName = Controls.get("formatName", "")
    # imagefile = ImgObj.data['data'][1]
    # if isinstance(imagefile, tuple) or isinstance(imagefile, list):
    #    imagefile, ImageTag =  imagefile # fix for multiimage files
    # else:
    #    ImageTag = None # single-image file
    # sumImg = G2fil.RereadImageData(ImageReaderlist,imagefile,ImageTag=ImageTag,FormatName=formatName)
    sumImg = img["image"]
    # if sumImg is None:
    #    return []
    sumImg = np.array(sumImg, dtype="int32")
    darkImg = False
    if "dark image" in Controls:
        darkImg, darkScale = Controls["dark image"]
        if darkImg:
            # dImgObj = proj.image(darkImg)
            # formatName = dImgObj.data['Image Controls'].get('formatName','')
            # imagefile = dImgObj.data['data'][1]
            # if type(imagefile) is tuple:
            #    imagefile,ImageTag  = imagefile
            # darkImage = G2fil.RereadImageData(ImageReaderlist,imagefile,ImageTag=ImageTag,FormatName=formatName)
            # darkImage = load_image_tifffile(darkImg)
            darkImage = load_image(darkImg)
            if darkImg is None:
                raise Exception("Error reading dark image {}".format(darkImg))
            sumImg += np.array(darkImage * darkScale, dtype="int32")
    if "background image" in Controls:
        backImg, backScale = Controls["background image"]
        if backImg:  # ignores any transmission effect in the background image
            # bImgObj = proj.image(backImg)
            # formatName = bImgObj.data['Image Controls'].get('formatName','')
            # imagefile = bImgObj.data['data'][1]
            # ImageTag = None # fix this for multiimage files
            # backImage = G2fil.RereadImageData(ImageReaderlist,imagefile,ImageTag=ImageTag,FormatName=formatName)
            # backImage = load_image_tifffile(backImg)
            backImage = load_image(backImg)
            if backImage is None:
                raise Exception("Error reading background image {}".format(backImg))
            if darkImg:
                backImage += np.array(darkImage * darkScale / backScale, dtype="int32")
            else:
                sumImg += np.array(backImage * backScale, dtype="int32")
    if "Gain map" in Controls:
        gainMap = Controls["Gain map"]
        if gainMap:
            # gImgObj = proj.image(gainMap)
            # formatName = gImgObj.data['Image Controls'].get('formatName','')
            # imagefile = gImgObj.data['data'][1]
            # ImageTag = None # fix this for multiimage files
            # GMimage = G2fil.RereadImageData(ImageReaderlist,imagefile,ImageTag=ImageTag,FormatName=formatName)
            # GMimage = load_image_tifffile(gainMap)
            GMimage = load_image(gainMap)
            if GMimage is None:
                raise Exception("Error reading Gain map image {}".format(gainMap))
            sumImg = sumImg * GMimage / 1000
    sumImg -= int(Controls.get("Flat Bkg", 0))
    Imax = np.max(sumImg)
    Controls["range"] = [(0, Imax), [0, Imax]]
    corrected_image = np.asarray(sumImg, dtype="int32")
    img["corrected_image"] = corrected_image
    return corrected_image


# nonzeromask
def nonzeromask(image, mask_negative=True):
    if mask_negative:
        # 1 if positive, 0 if zero or negative
        nonzeromask = image > 0
    else:
        # 1 if not zero, 0 if exactly zero
        nonzeromask = np.array(image, dtype=bool)
    return nonzeromask


# polar correction
def pol_correct(image, polmap):
    image_p = np.array(image)
    image_p = np.array(
        image_p / polmap, dtype=np.int32
    )  # polmap showing high values in center column dropping to low values at right/left edges
    return image_p


# flat-field correction
def flatfield_correct(image, flatfield):
    image_f = np.array(image)
    image_f = np.array(image_f / flatfield, dtype=np.int32)
    return image_f


# G2sc
def GeneratePixelMask(
    img,
    esdMul=3.0,
    ttmin=0.0,
    ttmax=180.0,
    FrameMask=None,
    ThetaMap=None,
    fastmode=True,
    combineMasks=False,
):
    """Generate a Pixel mask with True at the location of pixels that are
    statistical outliers (in comparison with others with the same 2theta
    value.) The process for this is that a median is computed for pixels
    within a small 2theta window and then the median difference is computed
    from magnitude of the difference for those pixels from that median. The
    medians are used for this rather than a standard deviation as the
    computation used here is less sensitive to outliers.
    (See :func:`GSASIIimage.AutoPixelMask` and
    :func:`scipy.stats.median_abs_deviation` for more details.)

    Mask is placed into the G2image object where it will be
    accessed during integration. Note that this increases the .gpx file
    size significantly; use :meth:`~G2Image.clearPixelMask` to delete
    this if it need not be saved.

    This code is based on :func:`GSASIIimage.FastAutoPixelMask`
    but has been modified to recycle expensive computations
    where possible.

    :param float esdMul: Significance threshold applied to remove
        outliers. Default is 3. The larger this number, the fewer
        "glitches" that will be removed.
    :param float ttmin: A lower 2theta limit to be used for pixel
        searching. Pixels outside this region may be considered for
        establishing the medians, but only pixels with 2theta >= :attr:`ttmin`
        are masked. Default is 0.
    :param float ttmax: An upper 2theta limit to be used for pixel
        searching. Pixels outside this region may be considered for
        establishing the medians, but only pixels with 2theta < :attr:`ttmax`
        are masked. Default is 180.
    :param np.array FrameMask: An optional precomputed Frame mask
        (from :func:`~G2Image.MaskFrameMask`). Compute this once for
        a series of similar images to reduce computational time.
    :param np.array ThetaMap: An optional precomputed array that
        defines 2theta for each pixel, computed in
        :func:`~G2Image.MaskThetaMap`. Compute this once for
        a series of similar images to reduce computational time.
    :param bool fastmode: If True (default) fast Pixel map
        searching is done if the C module is available. If the
        module is not available or this is False, the pure Python
        implementatruion is used. It is not clear why False is
        ever needed.
    :param bool combineMasks: When True, the current Pixel mask
        will be combined with any previous Pixel map. If False (the
        default), the Pixel map from the current search will
        replace any previous ones. The reason for use of this as
        True would be where different :attr:`esdMul` values are
        used for different regions of the image (by setting
        :attr:`ttmin` & :attr:`ttmax`) so that the outlier level
        can be tuned by combining different searches.
    """
    import math

    sind = lambda x: math.sin(x * math.pi / 180.0)
    if img["corrected_image"] is not None:
        Image = img["corrected_image"]
    else:
        Image = _getCorrImage(img)
    Controls = img["Image Controls"]
    Masks = img["Masks"]
    if FrameMask is None:
        frame = Masks["Frames"]
        tam = np.ma.make_mask_none(Image.shape)
        if frame:
            tam = np.ma.mask_or(tam, MakeFrameMask(Controls, frame))
    else:
        tam = FrameMask
    if ThetaMap is None:
        TA = Make2ThetaAzimuthMap(Controls, (0, Image.shape[0]), (0, Image.shape[1]))[0]
    else:
        TA = ThetaMap
    # if PolMap is None:
    #    PolMap = G2img.Make2ThetaAzimuthMap(Controls,(0,Image.shape[0]),(0,Image.shape[1]))[3]
    # print(Image.shape, Image.dtype, Image.strides) #(2880,2880) int32 (11520,4)
    # Image = np.array(Image/PolMap,dtype=np.int32) #changing dtype to int32 floors the values. Needed for use with fmask, which expects specific dtype (though oddly specifies it needs np.float64, stride 1).
    # print(Image.shape, Image.dtype, Image.strides)
    LUtth = np.array(Controls["IOtth"])
    wave = Controls["wavelength"]
    dsp0 = wave / (2.0 * sind(LUtth[0] / 2.0))
    dsp1 = wave / (2.0 * sind(LUtth[1] / 2.0))
    x0 = GetDetectorXY2(dsp0, 0.0, Controls)[0]
    x1 = GetDetectorXY2(dsp1, 0.0, Controls)[0]
    if not np.any(x0) or not np.any(x1):
        raise Exception
    numChans = int(1000 * (x1 - x0) / Controls["pixelSize"][0]) // 2

    import fmask

    # G2fil.G2Print(f'Fast mask: Spots greater or less than {esdMul:.1f} of median abs deviation are masked')
    outMask = np.zeros_like(tam, dtype=bool).ravel()
    TThs = np.linspace(LUtth[0], LUtth[1], numChans, False)
    try:
        masked = fmask.mask(
            esdMul, tam.ravel(), TA.ravel(), Image.ravel(), TThs, outMask, ttmin, ttmax
        )
    except Exception as msg:
        print("Exception in fmask.mask\n\t", msg)
        raise Exception(msg)
    outMask = outMask.reshape(Image.shape)

    if Masks["SpotMask"].get("spotMask") is not None and combineMasks:
        Masks["SpotMask"]["spotMask"] |= outMask
    else:
        Masks["SpotMask"]["spotMask"] = outMask


# G2img; used in GeneratePixelMask
def GetDetectorXY2(dsp, azm, data):
    """Get detector x,y position from d-spacing (dsp), azimuth (azm,deg)
    & image controls dictionary (data)
    it seems to be only used in plotting
    """
    elcent, phi, radii = GetEllipse(dsp, data)
    phi = data["rotation"] - 90.0  # to give rotation of major axis
    tilt = data["tilt"]
    dist = data["distance"]
    cent = data["center"]
    tth = 2.0 * npasind(data["wavelength"] / (2.0 * dsp))
    stth = npsind(tth)
    cosb = npcosd(tilt)
    if radii[0] > 0.0:
        sinb = npsind(tilt)
        tanb = nptand(tilt)
        fplus = dist * tanb * stth / (cosb + stth)
        fminus = dist * tanb * stth / (cosb - stth)
        zdis = (fplus - fminus) / 2.0
        rsqplus = radii[0] ** 2 + radii[1] ** 2
        rsqminus = radii[0] ** 2 - radii[1] ** 2
        R = rsqminus * npcosd(2.0 * azm - 2.0 * phi) + rsqplus
        Q = (
            np.sqrt(2.0)
            * radii[0]
            * radii[1]
            * np.sqrt(R - 2.0 * zdis**2 * npsind(azm - phi) ** 2)
        )
        P = 2.0 * radii[0] ** 2 * zdis * npcosd(azm - phi)
        radius = (P + Q) / R
        xy = np.array([radius * npcosd(azm), radius * npsind(azm)])
        xy += cent
    else:  # hyperbola - both branches (one is way off screen!)
        sinb = abs(npsind(tilt))
        tanb = abs(nptand(tilt))
        f = dist * tanb * stth / (cosb + stth)
        v = dist * (tanb + nptand(tth - abs(tilt)))
        delt = dist * stth * (1 + stth * cosb) / (sinb * cosb * (stth + cosb))
        ecc = (v - f) / (delt - v)
        R = radii[1] * (ecc**2 - 1) / (1 - ecc * npcosd(azm))
        if tilt > 0.0:
            offset = 2.0 * radii[1] * ecc + f  # select other branch
            xy = [-R * npcosd(azm) - offset, -R * npsind(azm)]
        else:
            offset = -f
            xy = [-R * npcosd(azm) - offset, R * npsind(azm)]
        xy = -np.array(
            [
                xy[0] * npcosd(phi) + xy[1] * npsind(phi),
                xy[0] * npsind(phi) - xy[1] * npcosd(phi),
            ]
        )
        xy += cent
    if data["det2theta"]:
        xy[0] += dist * nptand(
            data["det2theta"] + data["tilt"] * npsind(data["rotation"])
        )
    return xy


# G2img; used in GetDetectorXY2
def GetEllipse(dsp, data):
    """uses Dandelin spheres to find ellipse or hyperbola parameters from detector geometry
    as given in image controls dictionary (data) and a d-spacing (dsp)
    """
    cent = data["center"]
    tilt = data["tilt"]
    phi = data["rotation"]
    dep = data.get("DetDepth", 0.0)
    tth = 2.0 * npasind(data["wavelength"] / (2.0 * dsp))
    dist = data["distance"]
    dxy = peneCorr(tth, dep, dist)
    return GetEllipse2(tth, dxy, dist, cent, tilt, phi)


# G2img; called by GetEllipse
def GetEllipse2(tth, dxy, dist, cent, tilt, phi):
    """uses Dandelin spheres to find ellipse or hyperbola parameters from detector geometry
    on output
    radii[0] (b-minor axis) set < 0. for hyperbola

    """
    radii = [0, 0]
    stth = sind(tth)
    cosb = cosd(tilt)
    tanb = tand(tilt)
    tbm = tand((tth - tilt) / 2.0)
    tbp = tand((tth + tilt) / 2.0)
    sinb = sind(tilt)
    d = dist + dxy
    if tth + abs(tilt) < 90.0:  # ellipse
        fplus = d * tanb * stth / (cosb + stth)
        fminus = d * tanb * stth / (cosb - stth)
        vplus = d * (tanb + (1 + tbm) / (1 - tbm)) * stth / (cosb + stth)
        vminus = d * (tanb + (1 - tbp) / (1 + tbp)) * stth / (cosb - stth)
        radii[0] = (
            np.sqrt((vplus + vminus) ** 2 - (fplus + fminus) ** 2) / 2.0
        )  # +minor axis
        radii[1] = (vplus + vminus) / 2.0  # major axis
        zdis = (fplus - fminus) / 2.0
    else:  # hyperbola!
        f = d * abs(tanb) * stth / (cosb + stth)
        v = d * (abs(tanb) + tand(tth - abs(tilt)))
        delt = d * stth * (1.0 + stth * cosb) / (abs(sinb) * cosb * (stth + cosb))
        eps = (v - f) / (delt - v)
        radii[0] = -eps * (delt - f) / np.sqrt(eps**2 - 1.0)  # -minor axis
        radii[1] = eps * (delt - f) / (eps**2 - 1.0)  # major axis
        if tilt > 0:
            zdis = f + radii[1] * eps
        else:
            zdis = -f
    # NB: zdis is || to major axis & phi is rotation of minor axis
    # thus shift from beam to ellipse center is [Z*sin(phi),-Z*cos(phi)]
    elcent = [cent[0] + zdis * sind(phi), cent[1] - zdis * cosd(phi)]
    return elcent, phi, radii


# create and save TA[x] maps
# from savemaps.py by Wenqian Xu
def getmaps(
    cache, imctrlname, pathmaps, save=True
):  # fast integration using the same imctrl and mask
    # TA = G2img.Make2ThetaAzimuthMap(imctrls,(0,imctrls['size'][0]),(0,imctrls['size'][1]))    #2-theta array, 2880 according to detector pixel numbers
    imctrls = cache["Image Controls"]
    TA = Make2ThetaAzimuthMap(imctrls, (0, imctrls["size"][0]), (0, imctrls["size"][1]))
    if save:
        imctrlname = os.path.split(imctrlname)[1]
        path1 = os.path.join(pathmaps, imctrlname)

        im = Image.fromarray(TA[0])
        im.save(os.path.splitext(path1)[0] + "_2thetamap.tif")
        cache["pixelTAmap"] = TA[0]
        im = Image.fromarray(TA[1])
        im.save(os.path.splitext(path1)[0] + "_azmmap.tif")
        cache["pixelAzmap"] = TA[1]
        im = Image.fromarray(TA[2])
        im.save(os.path.splitext(path1)[0] + "_pixelsampledistmap.tif")
        cache["pixelsampledistmap"] = TA[2]
        im = Image.fromarray(TA[3])
        im.save(os.path.splitext(path1)[0] + "_polscalemap.tif")
        cache["polscalemap"] = TA[3]
        Qmap = get_Qmap(TA[0], imctrls["wavelength"])
        im = Image.fromarray(Qmap)
        im.save(os.path.splitext(path1)[0] + "_qmap.tif")
        cache["pixelQmap"] = Qmap
    return


def get_Qmap(Tmap, wavelength):
    return 4 * np.pi * np.sin(Tmap / 2 * np.pi / 180) / wavelength


def get_azimbands(azmap, numChansAzim):
    dazim = (360) / numChansAzim
    azimband = np.array(azmap / dazim, dtype=np.int32)
    return azimband


# G2img; used in Make2ThetaAzimuthMap
def GetTthAzmG(x, y, data):
    """Give 2-theta, azimuth & geometric corr. values for detector x,y position;
    calibration info in data - only used in integration for detector 2-theta != 0.
    checked OK for ellipses & hyperbola
    This is the slow step in image integration
    """

    def costth(xyz):
        u = xyz / np.linalg.norm(xyz, axis=-1)[:, :, np.newaxis]
        return np.dot(u, np.array([0.0, 0.0, 1.0]))

    # zero detector 2-theta: tested with tilted images - perfect integrations
    dx = x - data["center"][0]
    dy = y - data["center"][1]
    tilt = data["tilt"]
    dist = data["distance"] / npcosd(tilt)  # sample-beam intersection point
    T = makeMat(tilt, 0)
    R = makeMat(data["rotation"], 2)
    MN = np.inner(R, np.inner(R, T))
    dxyz0 = np.inner(
        np.dstack([dx, dy, np.zeros_like(dx)]), MN
    )  # correct for 45 deg tilt
    dxyz0 += np.array([0.0, 0.0, dist])
    if data["DetDepth"]:
        ctth0 = costth(dxyz0)
        tth0 = npacosd(ctth0)
        dzp = peneCorr(tth0, data["DetDepth"], dist)
        dxyz0[:, :, 2] += dzp
    # non zero detector 2-theta:
    if data.get("det2theta", 0):
        tthMat = makeMat(data["det2theta"], 1)
        dxyz = np.inner(dxyz0, tthMat.T)
    else:
        dxyz = dxyz0
    ctth = costth(dxyz)
    tth = npacosd(ctth)
    azm = (npatan2d(dxyz[:, :, 1], dxyz[:, :, 0]) + data["azmthOff"] + 720.0) % 360.0
    # G-calculation
    x0 = data["distance"] * nptand(tilt)
    x0x = x0 * npcosd(data["rotation"])
    x0y = x0 * npsind(data["rotation"])
    distsq = data["distance"] ** 2
    G = (
        (dx - x0x) ** 2 + (dy - x0y) ** 2 + distsq
    ) / distsq  # for geometric correction = 1/cos(2theta)^2 if tilt=0.
    return tth, azm, G


# G2img; used in Make2ThetaAzimuthMap
def GetTthAzmG2(x, y, data):
    """Give 2-theta, azimuth & geometric corr. values for detector x,y position;
    calibration info in data - only used in integration for detector 2-theta = 0
    """
    tilt = data["tilt"]
    dist = data["distance"] / npcosd(tilt)
    MN = -np.inner(makeMat(data["rotation"], 2), makeMat(tilt, 0))
    dx = x - data["center"][0]
    dy = y - data["center"][1]
    dz = np.dot(np.dstack([dx.T, dy.T, np.zeros_like(dx.T)]), MN).T[2]
    xyZ = dx**2 + dy**2 - dz**2
    tth0 = npatand(np.sqrt(xyZ) / (dist - dz))
    dzp = peneCorr(tth0, data["DetDepth"], dist)
    tth = npatan2d(np.sqrt(xyZ), dist - dz + dzp)
    azm = (npatan2d(dy, dx) + data["azmthOff"] + 720.0) % 360.0

    distsq = data["distance"] ** 2
    x0 = data["distance"] * nptand(tilt)
    x0x = x0 * npcosd(data["rotation"])
    x0y = x0 * npsind(data["rotation"])
    G = (
        (dx - x0x) ** 2 + (dy - x0y) ** 2 + distsq
    ) / distsq  # for geometric correction = 1/cos(2theta)^2 if tilt=0.
    return tth, azm, G


# G2img; used in GetTthAzmG and GetTthAzmG2, used for Make2ThetaAzimuthMap
def makeMat(Angle, Axis):
    """Make rotation matrix from Angle and Axis

    :param float Angle: in degrees
    :param int Axis: 0 for rotation about x, 1 for about y, etc.
    """
    cs = npcosd(Angle)
    ss = npsind(Angle)
    M = np.array(([1.0, 0.0, 0.0], [0.0, cs, -ss], [0.0, ss, cs]), dtype=np.float32)
    return np.roll(np.roll(M, Axis, axis=0), Axis, axis=1)


# G2img; used in GetEllipse (GeneratePixelMask), GetTthAzmG, GetTthAzmG2 (Make2ThetaAzimuthMap)
def peneCorr(tth, dep, dist):
    "Needs a doc string"
    return dep * (1.0 - npcosd(tth)) * dist**2 / 1000.0  # best one


def Qmap(Tmap, wavelength):
    return 4 * np.pi * np.sin(Tmap / 2 * np.pi / 180) / wavelength


def prepare_qmaps(tth_map, pol_map, dist_map, tth_min, tth_max, numChans):
    tth = tth_map.ravel()
    raveled_pol = pol_map.ravel()
    raveled_dist = dist_map.ravel()

    tth_delta = (tth_max - tth_min) / numChans
    tth_list = np.arange(tth_min, tth_max + tth_delta / 2.0, tth_delta)
    tth_val = ((tth_list[1:] + tth_list[:-1]) / 2.0).astype(np.float32)

    tth_idx = np.zeros_like(tth, dtype=np.int32)
    roi_1 = np.zeros_like(tth, dtype=bool)

    for idx, val in enumerate(tth_list[1:]):
        roi_2 = tth < val
        tth_idx[(~roi_1) * roi_2] = idx + 1
        roi_1[:] = roi_2

    tth_idx = torch.from_numpy(tth_idx)
    tth_val = torch.from_numpy(tth_val)
    return tth_idx, tth_val, raveled_pol, raveled_dist, len(tth_val)


def pytorch_integrate(
    image, mask, tth_idx, tth_val, raveled_pol, raveled_dist, tth_size
):
    data = image.ravel()
    data = torch.from_numpy(data)
    mask = mask.ravel()
    mask = ~mask
    mask = torch.from_numpy(mask)
    data = (
        data / raveled_pol * raveled_dist**1.5
    )  # the dist map is squared distance, here it needs to be cubic

    # no masked-array option for torch.bincount or np.bincount
    # val = torch.bincount(tth_idx, weights=data*mask, minlength=tth_size)[1:]
    masked_tth_idx = tth_idx * mask
    val = torch.bincount(masked_tth_idx, weights=data, minlength=tth_size)[1:]
    norm_factor = torch.clamp(torch.bincount(masked_tth_idx)[1:], min=1, max=None)
    val /= norm_factor
    data = torch.vstack([tth_val, val]).numpy().T

    return data


def r_and_phi_hat(image_shape, center):
    pixels = np.indices(image_shape)
    a = np.array([center[0], center[1]])
    b = np.ones(image_shape)
    centers = a[:, None, None] * b
    displacements = pixels - centers
    norms = np.linalg.norm(displacements, axis=0)
    r_hat = displacements / norms
    a = np.array([1, -1])
    temp = a[:, None, None] * b
    phi_hat = np.multiply(r_hat[::-1, :, :], temp)
    return r_hat, phi_hat


def gradient_cache(image_shape, center, footprint):
    # calculate distances and x-y-basis angles once for each pixel in footprint
    t0 = time.time()
    if not all([i % 2 == 1 for i in footprint.shape]):
        raise ValueError("Footprint shape must be odd in each direction.")
    central_footprint_point = np.array([i // 2 for i in footprint.shape])
    # print("central footprint point:", central_footprint_point)
    footprint[central_footprint_point[0], central_footprint_point[1]] = 0
    # print("footprint")
    # print(footprint)
    distances = np.zeros(footprint.shape)
    direction_vectors = np.zeros((2, footprint.shape[0], footprint.shape[1]))
    rel_coords = np.indices(footprint.shape) - central_footprint_point[
        :, None, None
    ] * np.ones_like(footprint)
    for i, j in np.ndindex(distances.shape):
        if footprint[i, j] != 0:
            distances[i, j] = np.sqrt(
                (i - central_footprint_point[0]) ** 2
                + (j - central_footprint_point[1]) ** 2
            )
            direction_vectors[:, i, j] = rel_coords[:, i, j] / np.linalg.norm(
                rel_coords[:, i, j], axis=0
            )

    # Let p = current pixel and center of window, q = neighbor, x = x_dots, d = full distance
    # Let g be the 1st order derivs using each point and q. Center is 0.
    # g = fx(q-p)/d for all f nonzero, else 0. f = footprint weight, whether 1, 0, or even 2, 1.5, etc.
    # Need each position in kernel to only be a multiple of q.
    # g = fxq/d - fxp/d
    # Let grad = output gradient at p
    # grad = (1/sum(f))sum(g)
    # grad = (1/sum(f))sum(fxq/d) - (nonzero(f)/sum(f))sum(fxp/d)
    # now we can make a kernel out of this
    # most terms will be (1/sum(f))fx/d
    # central term will be long: -(nonzero(f)/sum(f))sum(fx/d)
    x_dots = direction_vectors[1]
    y_dots = direction_vectors[0]
    sum_footprint_x = np.sum(footprint[np.nonzero(footprint * x_dots)])
    sum_footprint_y = np.sum(footprint[np.nonzero(footprint * y_dots)])
    nonzero_footprint_x = np.sum(
        np.ones_like(footprint)[np.nonzero(footprint * x_dots)]
    )
    nonzero_footprint_y = np.sum(
        np.ones_like(footprint)[np.nonzero(footprint * y_dots)]
    )
    kernel_x = np.zeros_like(distances)
    kernel_y = np.zeros_like(distances)
    # much shorter for loop
    for i in range(footprint.shape[0]):
        for j in range(footprint.shape[1]):
            if (i == central_footprint_point[0]) and (j == central_footprint_point[1]):
                kernel_x[i, j] = -(nonzero_footprint_x / sum_footprint_x) * np.sum(
                    footprint * x_dots / np.where(distances == 0, 999, distances)
                )  # distances should only be zero at center, where footprint = 0 anyway
                kernel_y[i, j] = -(nonzero_footprint_y / sum_footprint_y) * np.sum(
                    footprint * y_dots / np.where(distances == 0, 999, distances)
                )
            elif footprint[i, j] == 0:
                kernel_x[i, j] = 0
                kernel_y[i, j] = 0
            else:
                kernel_x[i, j] = (
                    (1 / sum_footprint_x)
                    * footprint[i, j]
                    * x_dots[i, j]
                    / distances[i, j]
                )
                kernel_y[i, j] = (
                    (1 / sum_footprint_x)
                    * footprint[i, j]
                    * y_dots[i, j]
                    / distances[i, j]
                )
    # print("central kernel calculation")
    # print("Kernels calculated, getting convolutions")
    r_hat, phi_hat = r_and_phi_hat(image_shape, center)
    t1 = time.time()
    print(
        "Gradient time spent on cache calculations: {0:.2f}s".format(
            t1 - t0
        )
    )
    return_dict = {
        "r_hat": r_hat,
        "phi_hat": phi_hat,
        "kernel_x": kernel_x,
        "kernel_y": kernel_y,
    }
    return return_dict


def radial_and_azim_gradient(
    image, r_hat, phi_hat, kernel_x, kernel_y, azim_only=False
):
    # footprint = footprint.astype(np.uint)

    from scipy.ndimage import correlate

    grad_x = correlate(image, kernel_x)
    grad_y = correlate(image, kernel_y)
    # print("Convolutions done")

    grad = np.stack([grad_y, grad_x], axis=0)
    azim_grad = np.einsum("ijk,ijk -> jk", grad, phi_hat)
    if azim_only:
        return azim_grad
    else:
        radial_grad = np.einsum("ijk,ijk -> jk", grad, r_hat)
        return radial_grad, azim_grad


def qwidth_area_classification(
    om,
    Qmap,
    azmap,
    min_arc_area=100,
    Q_max=0.1,
    azim_min=3.5,
    compare_shape=True,
    area_Q_shape_min=150000,
    azim_Q_shape_min=100,
    return_time=False,
):
    # azim_Q: 150
    # area_Q: 4500
    # area_Q^2: 350000
    if return_time:
        time0 = time.time()
    om = np.array(om)
    labeled_mask = ski.measure.label(om)
    # print("Labeled mask values at Amine locations: {0},{1}".format(labeled_mask[1675,2155],labeled_mask[1539,1189])) #18178
    # print("Labeled mask values at Ties locations: {0},{1}".format(labeled_mask[2365,1580],labeled_mask[2328,1537]))

    props = ["label", "area", "intensity_min", "intensity_max"]

    props_table = ski.measure.regionprops_table(
        labeled_mask, intensity_image=Qmap, properties=props
    )
    props_table = pd.DataFrame(props_table)
    props_table["diffs_Q"] = props_table["intensity_max"] - props_table["intensity_min"]
    props_table_azim = ski.measure.regionprops_table(
        labeled_mask,
        intensity_image=azmap,
        properties=["label", "intensity_min", "intensity_max"],
    )
    props_table_azim = pd.DataFrame(props_table_azim)
    props_table["diffs_azim"] = (
        props_table_azim["intensity_max"] - props_table_azim["intensity_min"]
    )  # same label should mean same order, same index

    # median absolute deviations sorted by label
    # Min arc area is 100, so only need to look at this value for those anyway
    small_clusters = props_table.loc[:, "area"].values < min_arc_area
    small_cluster_indices = props_table.iloc[small_clusters].index
    props_table.drop(small_cluster_indices, axis=0, inplace=True)

    # recalculate azim difference for those crossing the azimuth; min will be ~0 and max ~359
    # reorganize w/o for loops
    for label in props_table.loc[props_table["diffs_azim"] > 359, "label"].values:
        values = np.array(azmap[labeled_mask == label])
        values.sort()
        diffs = values[1:] - values[:-1]
        # find diffs > 10
        location = np.argwhere(diffs > 10)[0]
        new_diff = 360 - (values[location + 1] - values[location])
        props_table.loc[props_table["label"] == label, "diffs_azim"] = new_diff

    # recalculate Q difference based on 10th,90th percentile values
    for label in props_table["label"].values:
        values = np.array(Qmap[labeled_mask == label])
        values.sort()
        Qmin, Qmax = np.percentile(values, [10, 90])
        props_table.loc[props_table["label"] == label, "diffs_Q"] = Qmax - Qmin

    # print(props_table['area'])
    # props_table['mad2_vs_area'] = props_table['diffs'].values**2 / props_table['area'].values
    props_table["area_over_width"] = props_table["area"] / props_table["diffs_Q"]
    props_table["area_over_width2"] = props_table["area"] / (
        props_table["diffs_Q"] ** 2
    )
    props_table["azim_vs_Q"] = props_table["diffs_azim"] / props_table["diffs_Q"]

    # print(props_table.loc[props_table['label'] == 18178,:])
    # print(props_table.loc[props_table['label'] == 16621,:])
    # print(props_table.loc[props_table['label'] == 24661,:])
    # print(props_table.loc[props_table['label'] == 26275,:])
    # print(props_table)

    arcs_bool = props_table.loc[:, "diffs_Q"].values < Q_max
    arcs_bool = np.logical_and(
        props_table.loc[:, "diffs_azim"].values > azim_min, arcs_bool
    )
    if compare_shape:
        # arcs_bool = np.logical_and(props_table.loc[:,'mad2_vs_area'].values <= shape_max, arcs_bool)
        # arcs_bool = np.logical_and(props_table.loc[:,'area_over_width'].values > area_Q_shape_min, arcs_bool)
        arcs_bool = np.logical_and(
            props_table.loc[:, "area_over_width2"].values > area_Q_shape_min, arcs_bool
        )
        # arcs_bool = np.logical_and(props_table.loc[:,'azim_vs_Q'].values > azim_Q_shape_min, arcs_bool)
    # arc_clusters = props_table.loc[arcs,'label'].values
    arcs = props_table.iloc[arcs_bool]
    # arc_cluster_indices = props_table.iloc[arcs_bool].index
    arc_cluster_indices = arcs["label"].values  # using index is one off

    # spot_mask = np.in1d(new_mask,spot_clusters)
    arc_mask = np.in1d(labeled_mask, arc_cluster_indices)
    # spot_mask = np.reshape(spot_mask,om.shape)
    arc_mask = np.reshape(arc_mask, om.shape)
    spot_mask = np.logical_and(om, ~arc_mask)

    if return_time:
        time1 = time.time()
        return spot_mask, arc_mask, time1 - time0
    else:
        return spot_mask, arc_mask


def remove_overlaps(labeled_cuts, predef_mask):
    labels = np.unique(labeled_cuts)
    # if this is just an array of 0, the only unique number will be 0 and we can stop
    if labels.shape == (1,):
        return labeled_cuts
    else:
        # discard 0
        labels = labels[1:]
        new_labels = []
        # find which labels overlap
        for label in labels:
            overlaps = np.max(np.logical_and(labeled_cuts == label, predef_mask))
            if not overlaps:
                new_labels.append(label)
        new_cuts = np.zeros_like(labeled_cuts)
        for label in new_labels:
            new_cuts |= labeled_cuts == label
        return new_cuts


def split_grad_with_Q(
    image,
    om,
    arc_mask,
    gradient_dict,
    qmap,
    azmap,
    threshold_percentile=0.1,
    return_grad=False,
    return_partials=False,
    interpolate=False,
    predef_mask=None,
    predef_mask_extended=None,
):
    if interpolate:
        if predef_mask is None:
            print("No predefined mask provided. Skipping interpolation.")
        else:
            from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

            kernel = Gaussian2DKernel(x_stddev=1)
            image[predef_mask] = np.nan
            interpolated_image = interpolate_replace_nans(image, kernel)
            radial_grad, azim_grad = radial_and_azim_gradient(
                interpolated_image,
                gradient_dict["r_hat"],
                gradient_dict["phi_hat"],
                gradient_dict["kernel_x"],
                gradient_dict["kernel_y"],
            )
            azim_grad_2 = radial_and_azim_gradient(
                azim_grad,
                gradient_dict["r_hat"],
                gradient_dict["phi_hat"],
                gradient_dict["kernel_x"],
                gradient_dict["kernel_y"],
                azim_only=True,
            )
            radial_grad_2, _ = radial_and_azim_gradient(
                radial_grad,
                gradient_dict["r_hat"],
                gradient_dict["phi_hat"],
                gradient_dict["kernel_x"],
                gradient_dict["kernel_y"],
            )
    else:
        radial_grad, azim_grad = radial_and_azim_gradient(
            image,
            gradient_dict["r_hat"],
            gradient_dict["phi_hat"],
            gradient_dict["kernel_x"],
            gradient_dict["kernel_y"],
        )
        azim_grad_2 = radial_and_azim_gradient(
            azim_grad,
            gradient_dict["r_hat"],
            gradient_dict["phi_hat"],
            gradient_dict["kernel_x"],
            gradient_dict["kernel_y"],
            azim_only=True,
        )
        radial_grad_2, _ = radial_and_azim_gradient(
            radial_grad,
            gradient_dict["r_hat"],
            gradient_dict["phi_hat"],
            gradient_dict["kernel_x"],
            gradient_dict["kernel_y"],
        )

    # use a percentile of the radial gradient rather than the azimuthal gradient:
    # there will always be rings, but we cannot assume an amount of spots. azim could be near zero.
    threshold = np.percentile(radial_grad_2, threshold_percentile)
    on_arc_threshold = np.percentile(radial_grad_2, 10)

    # testing of potential arc sections
    labeled_mask = ski.measure.label(arc_mask)
    # finding whether the center line lies on an arc
    # props = ['label','centroid','coords']
    # props = ['label','centroid','slice']
    props = ["label", "intensity_min", "intensity_max"]
    props_table = ski.measure.regionprops_table(
        labeled_mask, properties=props, intensity_image=qmap
    )  # replace with array of label values
    # props_table = {'label':np.arange(1,np.max(labeled_mask)+1)}
    props_table = pd.DataFrame(props_table)
    # print(props_table.loc[props_table['label']==9,'coords'].values)
    # print(props_table.loc[props_table['label']==9,'slice'].values)
    # props_table['Q_center'] = [qmap[int(centroid_0), int(centroid_1)] for centroid_0, centroid_1 in zip(props_table['centroid-0'],props_table['centroid-1'])]
    props_table["Q_median"] = [
        np.median(qmap[labeled_mask == label]) for label in props_table["label"]
    ]
    # check values of second radial derivative in a range near that q value
    # props_table['central_coords'] = [coords[np.abs(qmap[tuple(np.moveaxis(coords,1,0))] - center) < 0.02] for coords, center in zip(props_table['coords'],props_table['Q_center'])]
    props_table["central_slice"] = [
        qmap[labeled_mask == label] - center < 0.02
        for label, center in zip(props_table["label"], props_table["Q_median"])
    ]
    # print(props_table[['coords','central_coords','centroid-0','centroid-1']])
    # print(props_table[['slice','central_slice']])
    # props_table['second_radial'] = [radial_grad_2[tuple(np.moveaxis(coords,1,0))] for coords in props_table['central_coords']]
    props_table["second_radial"] = [
        radial_grad_2[labeled_mask == label][central_slice]
        for label, central_slice in zip(
            props_table["label"], props_table["central_slice"]
        )
    ]
    # print(props_table[['coords','second_radial']])
    # print(props_table[['Q_median','central_slice','second_radial']])
    props_table["radial_grad_percentile"] = [
        np.percentile(radial_grads, 20) for radial_grads in props_table["second_radial"]
    ]
    props_table["on_arc"] = props_table["radial_grad_percentile"] < on_arc_threshold
    props_table["diffs"] = props_table["intensity_max"] - props_table["intensity_min"]
    # props_table['not_on_arc'] = ~props_table['on_arc']
    # print(props_table[['second_radial','radial_grad_percentile','not_on_arc']])
    # print(props_table[['second_radial','radial_grad_percentile','on_arc']])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(labeled_mask == 9,origin='lower')
    # plt.show()

    # small_clusters = props_table.loc[:,'area'].values < min_arc_area
    # small_cluster_indices = props_table.iloc[small_clusters].index
    # props_table.drop(small_cluster_indices,axis=0,inplace=True)
    props_table_shortened = props_table.drop(
        props_table.iloc[~props_table["on_arc"].values].index, axis=0
    )
    # print(props_table_shortened[['second_radial','radial_grad_percentile','on_arc']])
    arc_cluster_indices = props_table_shortened[
        "label"
    ].values  # using index is one off

    # spot_mask = np.in1d(new_mask,spot_clusters)
    Qgrad_arc_mask = np.in1d(labeled_mask, arc_cluster_indices)
    # spot_mask = np.reshape(spot_mask,om.shape)
    Qgrad_arc_mask = np.reshape(Qgrad_arc_mask, om.shape)

    gradient_mask = azim_grad_2 < threshold
    # only look at the areas overlapping potential arcs
    gradient_mask = np.logical_and(gradient_mask, Qgrad_arc_mask)
    # if there aren't any clusters, just skip this part entirely. Maximum will be False if nothing is there.
    if np.max(gradient_mask) > 0:
        labeled_gradients = ski.measure.label(gradient_mask)
        gradient_props = ["label", "centroid", "coords", "area"]
        gradient_props_table = ski.measure.regionprops_table(
            labeled_gradients, properties=gradient_props
        )
        gradient_props_table = pd.DataFrame(gradient_props_table)

        # # making circles
        # props_table['ext_centroids'] = [np.ones((int(a)))[:,None]*np.array([centroid_0,centroid_1]) for centroid_0,centroid_1,a in zip(props_table['centroid-0'],props_table['centroid-1'],props_table['area'])]
        # # now need to get the arrays of [[a1,b1],[a2,b2],...] - [a,b]:
        # props_table['radii'] = [np.max(coords - centroids) for coords, centroids in zip(props_table['coords'],props_table['ext_centroids'])]

        # # now to make a new mask using centroid-0, centroid-1, and radii
        # props_table['rr_cc'] = [ski.draw.disk((centroid_0,centroid_1),radius+1,shape=image.shape) for centroid_0,centroid_1,radius in zip(props_table['centroid-0'],props_table['centroid-1'],props_table['radii'])]
        # cuts = np.zeros(image.shape,dtype=bool)
        # for rr_cc in props_table['rr_cc'].values:
        #     cuts[rr_cc] = 1

        # find corresponding cluster label, width of cluster
        gradient_props_table["cluster_label"] = [
            labeled_mask[int(centroid_0), int(centroid_1)]
            for centroid_0, centroid_1 in zip(
                gradient_props_table["centroid-0"], gradient_props_table["centroid-1"]
            )
        ]
        gradient_props_table["azim_centroid"] = [
            azmap[int(centroid_0), int(centroid_1)]
            for centroid_0, centroid_1 in zip(
                gradient_props_table["centroid-0"], gradient_props_table["centroid-1"]
            )
        ]
        # gradient_props_table['Qwidth'] = [props_table.loc[props_table['label']==label,'diffs'] for label in zip(gradient_props_table['cluster_label'])]
        # gradient_props_table['azim_width'] = gradient_props_table['Qwidth']*5

        # handle cases when centroid falls off the mask
        labels = gradient_props_table["cluster_label"].values
        # print(labels)
        vals = []
        for i in range(len(labels)):
            # For now, if centroid is not on the mask, set diff to max width
            if labels[i] == 0:
                diff = 0.1
            else:
                diff = props_table.loc[
                    props_table["label"] == labels[i], "diffs"
                ].values[0]
            vals.append(diff)
        vals = np.array(vals)
        gradient_props_table["Qwidth"] = vals
        gradient_props_table["azim_width"] = vals * 5

        # cut azim section of labeled_mask from centroid-azim_width to centroid+azim_width
        gradient_props_table["cuts"] = [
            np.logical_and(
                labeled_mask == label, modulo_range(azmap, center_azim, azim_width)
            )
            for label, center_azim, azim_width in zip(
                gradient_props_table["cluster_label"],
                gradient_props_table["azim_centroid"],
                gradient_props_table["azim_width"],
            )
        ]
        if predef_mask_extended is not None:
            gradient_props_table["labeled_cuts"] = [
                ski.measure.label(cuts) for cuts in gradient_props_table["cuts"]
            ]
            # if a labeled section overlaps with the extended predefined mask (or possibly just the predefined mask, since these take a range anyway), drop it from the cut sections
            gradient_props_table["new_cuts"] = [
                remove_overlaps(labeled_cuts, predef_mask_extended)
                for labeled_cuts in gradient_props_table["labeled_cuts"]
            ]
        # to_cut = np.zeros_like(om)
        # print(gradient_props_table[['cluster_label','azim_centroid','azim_width','cuts']])

        cuts = np.sum(gradient_props_table["new_cuts"].values, axis=0).astype(bool)

        # cut section of gradient mask from arc mask
        new_arc_mask = np.logical_and(Qgrad_arc_mask, ~cuts)
        new_spot_mask = np.logical_and(om, ~new_arc_mask)

    else:
        # print("No spots to cut out. Skipping last step.")
        new_arc_mask = Qgrad_arc_mask
        new_spot_mask = np.logical_and(om, ~new_arc_mask)

    # spots stats
    # only arcs have stayed in one table; spots have been scattered across multiple
    # so spots stats table needs to be recreated at this point
    spots_props = ["label", "area", "intensity_mean"]
    labeled_spots = ski.measure.label(new_spot_mask)
    spots_table = ski.measure.regionprops_table(
        labeled_spots, qmap, properties=spots_props
    )
    spots_table = pd.DataFrame(spots_table)

    to_return = [new_spot_mask, new_arc_mask, spots_table]

    if return_grad:
        to_return.append(radial_grad_2)
        to_return.append(azim_grad_2)

    if return_partials:
        to_return.append(Qgrad_arc_mask)

    return to_return


def modulo_range(array, center, range):
    diff = (array - center) % 360
    # range = range.values
    return np.logical_or(diff < range, diff > (360 - range))


def current_splitting_method(
    image,
    om,
    qmap,
    azmap,
    gradient_dict,
    threshold_percentile=0.1,
    return_steps=False,
    interpolate=False,
    predef_mask=None,
    predef_mask_extended=None,
):
    # t0 = time.time()
    # base_spot, base_arc = qwidth_area_classification(om, qmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_max = 0.00001)
    # base_spot, base_arc = qwidth_area_classification(om, qmap, azmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_min = 3500)
    base_spot, base_arc = qwidth_area_classification(
        om, qmap, azmap, min_arc_area=100, Q_max=0.08, azim_min=3.5, compare_shape=True
    )
    # t1 = time.time()
    # print("Time to do initial q width / area classification: {0:.2f}s".format(t1-t0))
    # spot_mask, arc_mask = split_grad(image,om,base_arc,gradient_dict,threshold_percentile=threshold_percentile)
    if return_steps:
        spot_mask, arc_mask, spots_table, radial_grad_2, azim_grad_2, qgrad_arc_mask = (
            split_grad_with_Q(
                image,
                om,
                base_arc,
                gradient_dict,
                qmap,
                azmap,
                threshold_percentile=threshold_percentile,
                return_partials=True,
                return_grad=True,
                interpolate=interpolate,
                predef_mask=predef_mask,
                predef_mask_extended=predef_mask_extended,
            )
        )
    else:
        spot_mask, arc_mask, spots_table = split_grad_with_Q(
            image,
            om,
            base_arc,
            gradient_dict,
            qmap,
            azmap,
            threshold_percentile=threshold_percentile,
            interpolate=interpolate,
            predef_mask=predef_mask,
            predef_mask_extended=predef_mask_extended,
        )

    to_return = [spot_mask, arc_mask, spots_table]
    if return_steps:
        to_return.append(base_arc)
        to_return.append(qgrad_arc_mask)
        to_return.append(azim_grad_2)
        to_return.append(radial_grad_2)
    return to_return


class CacheCreator(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(
        self,
        cache,
        directory,
        filename,
        imctrlname,
        flatfield,
        imgmaskname,
        blkSize,
        logging=False,
    ):
        super().__init__()
        self.cache = cache
        self.directory = directory
        self.filename = filename
        self.imctrlname = imctrlname
        self.flatfield = flatfield
        self.imgmaskname = imgmaskname
        self.blkSize = blkSize
        self.logging = logging

    def run(self):
        cache_time = time.time()
        if self.logging:
            print("Creating cache")
        image_dict = read_image(self.filename)
        # img.loadControls(imctrlname)   # set controls/calibrations/masks
        with open(self.imctrlname, "r") as imctrlfile:
            lines = imctrlfile.readlines()
            LoadControls(lines, image_dict["Image Controls"])
        # cache['Image Controls'] = img.getControls() # save controls & masks contents for quick reload
        # self.cache['image'] = tf.imread(self.filename)
        self.cache["image"] = load_image(self.filename)
        predef_mask = {}
        if (self.imgmaskname is not None) and (self.imgmaskname != ""):
            # img.loadMasks(imgmaskname)
            suffix = self.imgmaskname.split(".")[1]
            if suffix == "immask":
                readMasks(self.imgmaskname, image_dict["Masks"], False)
            elif suffix == "tif":
                print(self.imgmaskname)
                predef_mask = read_image(self.imgmaskname)
        else:
            predef_mask["image"] = np.zeros_like(image_dict["image"], dtype=bool)
        self.cache["predef_mask"] = predef_mask

        flatfield_image = np.ones_like(self.cache["image"])
        if (self.flatfield is not None) and (self.flatfield != ""):
            # flatfield_image = tf.imread(self.flatfield)
            flatfield_image = load_image(self.flatfield)
        self.cache["flatfield"] = flatfield_image

        # imctrlname = imctrlname.split("\\")[-1].split('/')[-1]
        # path1 =  os.path.join(pathmaps,imctrlname)
        # im = Image.fromarray(TA[0])
        # im.save(os.path.splitext(path1)[0] + '_2thetamap.tif')
        imsave = Image.fromarray(predef_mask["image"])
        imsave.save(
            os.path.join(
                self.directory,
                "maps",
                os.path.splitext(os.path.split(self.imctrlname)[1])[0] + "_predef.tif"
            )
        )
        imsave = Image.fromarray(flatfield_image)
        imsave.save(
            os.path.join(
                self.directory,
                "maps",
                os.path.splitext(os.path.split(self.imctrlname)[1])[0] + "_flatfield.tif"
            )
        )
        self.cache["Image Controls"] = image_dict["Image Controls"]
        # TODO: Look at image size?
        # img.setControl('pixelSize',[150.0,150.0])
        image_dict["Image Controls"]["pixelSize"] = [150.0, 150.0]
        self.cache["Image Controls"]["pixelSize"] = [150.0, 150.0]
        # cache['Masks'] = img.getMasks()
        # self.cache['Masks'] = image_dict['Masks']
        # cache['intMaskMap'] = img.IntMaskMap() # calc mask & TA arrays to save for integrations
        # for k,v in img_copy['Image Controls'].items():
        #    print(k)
        # for k in img.data['Image Controls'].keys():
        #    if k not in img_copy['Image Controls'].keys():
        #        print(k, img.data['Image Controls'][k])
        # Missing 5: size, samplechangerpos, det2theta, ImageTag, formatName
        # [2880,2880] None 0.0 None GSAS-II known TIF image
        # only size seems to be holding anything meaningful at this time, though det2theta and samplechangerpos could hold something later

        # self.cache["intMaskMap"] = MakeUseMask(
        #     image_dict["Image Controls"], image_dict["Masks"], blkSize=self.blkSize
        # )
        # cache['intTAmap'] = img.IntThetaAzMap()
        self.cache["intTAmap"] = MakeUseTA(image_dict["Image Controls"], self.blkSize)
        # cache['FrameMask'] = img.MaskFrameMask() # calc Frame mask & T array to save for Pixel masking
        self.cache["FrameMask"] = MaskFrameMask(image_dict)
        # cache['maskTmap'] = img.MaskThetaMap()
        self.cache["maskTmap"] = Make2ThetaAzimuthMap(
            image_dict["Image Controls"],
            (0, image_dict["Image Controls"]["size"][0]),
            (0, image_dict["Image Controls"]["size"][1]),
        )[0]
        getmaps(self.cache, self.imctrlname, os.path.join(self.directory,"maps"))
        # 2th fairly linear along center; calc 2th - pixelsize conversion
        center = self.cache["Image Controls"]["center"]
        center[0] = center[0] * 1000.0 / self.cache["Image Controls"]["pixelSize"][0]
        center[1] = center[1] * 1000.0 / self.cache["Image Controls"]["pixelSize"][1]
        self.cache["center"] = center
        image_dict["center"] = center

        # self.cache['d2th'] = (self.cache['pixelTAmap'][int(center[1]),0] - self.cache['pixelTAmap'][int(center[1]),99])/100
        self.cache["esdMul"] = 3
        numChansAzim = 360
        self.cache["azimband"] = get_azimbands(self.cache["pixelAzmap"], numChansAzim)

        # pytorch integration
        (
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        ) = prepare_qmaps(
            self.cache["pixelTAmap"],
            self.cache["polscalemap"],
            self.cache["pixelsampledistmap"],
            self.cache["Image Controls"]["IOtth"][0],
            self.cache["Image Controls"]["IOtth"][1],
            self.cache["Image Controls"]["outChannels"],
        )

        # comparisons
        # self.cache['Previous image'] = tf.imread(self.filename)
        # self.cache['First image'] = tf.imread(self.filename)
        self.cache["First image"] = load_image(self.filename)

        # gradient info
        self.cache["gradient"] = gradient_cache(
            predef_mask["image"].shape, center, np.ones((3, 3), dtype=np.uint)
        )

        self.cache["image_dict"] = image_dict

        cache_time = time.time() - cache_time
        print(cache_time)

        self.finished.emit()


class SingleIterator(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)

    def __init__(
        self,
        cache,
        filename,
        imctrlname,
        imgmaskname,
        directory,
        name,
        number,
        ext,
        closing_method="binary_closing",
        logging=False,
    ):
        super().__init__()
        self.cache = cache.copy()
        self.filename = filename
        self.imctrlname = imctrlname
        self.imgmaskname = imgmaskname
        self.directory = directory
        self.name = name
        self.number = number
        self.ext = ext
        self.closing_method = closing_method
        self.logging = logging

    def run(self):
        # load in the image
        single_iter_times = []
        time_checkpoints = []
        single_iter_times.append(time.time())
        # if logging:
        #     print("Loading image {0}".format(filename))
        # image = load_image_tifffile(filename)
        # create temporary project
        # single_iter_times.append(time.time())
        if self.logging:
            print("Creating project")
        # gpx = G2sc.G2Project(newgpx=PathWrap('integration.gpx'))
        # img = gpx.add_image(filename,fmthint="TIF",cacheImage=True)[0]
        # image_dict = read_image(filename)
        # Compare data cache from add_image to loadControls()
        # print(gpx.data)
        # print("Data before loading controls")
        # print(img.data)

        # read in and correct controls, cache

        if self.logging:
            print("Pulling from cache")
        # img.setControls(cache['Image Controls'])
        # image_dict['Image Controls'] = self.cache['Image Controls']
        image_dict = self.cache["image_dict"]
        # image_dict['image'] = tf.imread(self.filename)
        image_dict["image"] = load_image(self.filename)
        # add the correction in now
        image_dict["image"] = flatfield_correct(
            image_dict["image"], self.cache["flatfield"]
        )
        image_dict["corrected_image"] = None
        # img.setMasks(cache['Masks'],True)  # True: reset threshold masks
        # image_dict['Masks'] = self.cache['Masks']
        single_iter_times.append(time.time())
        time_checkpoints.append("Cache")
        # print("Data after loading controls")
        # print(img.data)
        # print(img.data['Image Controls'])
        # print(img.data['Comments'])
        time_checkpoints.append('Cache')
        # print("Data after loading controls")
        #print(img.data)
        #print(img.data['Image Controls'])
        #print(img.data['Comments'])
        print("Cache")
        # for k,v in cache.items():
        #    print(k,v)
        # get zero mask
        # Add in predefined mask to frame mask
        if self.logging:
            print("Adding in frame and nonzero mask to predefined mask")
        # mask_nonzero = nonzeromask(image_dict['image'])
        nonpositive_mask = ~nonzeromask(image_dict["image"], mask_negative=True)
        imsave = Image.fromarray(nonpositive_mask)
        imsave.save(
            os.path.join(
                self.directory,
                "masks",
                self.name + "-" + self.number + "_nonpositive.tif"
            )
        )
        # frame_and_predef = np.logical_or(self.cache['FrameMask'],self.cache['predef_mask'])
        # frame_and_predef = np.logical_or(frame_and_predef,~mask_nonzero)
        predef_and_nonpositive = np.logical_or(
            nonpositive_mask, self.cache["predef_mask"]["image"]
        )
        predef_mask_extended = ski.morphology.binary_dilation(
            predef_and_nonpositive, footprint=ski.morphology.square(7)
        )  # extend out by three pixels; use for determining whether something is nearby
        # print(mask_nonzero.dtype)
        # print(np.array(self.cache['predef_mask']['image']).dtype)
        # print(predef_mask)
        frame_and_predef = np.logical_or(
            predef_and_nonpositive, self.cache["FrameMask"]
        )
        # imsave = Image.fromarray(frame_and_predef)
        # imsave.save(self.directory + '\\masks\\' + self.name + '-' + self.number + '_predef.tif')
        single_iter_times.append(time.time())
        time_checkpoints.append("Frame and Predef masks")
        # polar-correct the image
        # if logging: print("Polar-correcting the image")
        # image_pol = pol_correct(image_dict['image'],self.cache['polscalemap'])
        # single_iter_times.append(time.time())
        # get outlier mask using polar-corrected image
        if self.logging:
            print("Generating pixel mask")
        # TODO: set image data to polar-corrected and back
        # img.GeneratePixelMask(esdMul=cache['esdMul'],FrameMask=cache['FrameMask'],ThetaMap=cache['maskTmap'])
        # GeneratePixelMask(image_dict,esdMul=self.cache['esdMul'],FrameMask=self.cache['FrameMask'],ThetaMap=self.cache['maskTmap'])
        # print("cache:",self.cache)
        # print("image dict:",image_dict)
        GeneratePixelMask(
            image_dict,
            esdMul=self.cache["esdMul"],
            FrameMask=frame_and_predef,
            ThetaMap=self.cache["maskTmap"],
        )
        # outlier_mask = img.data['Masks']['SpotMask']['spotMask']
        outlier_mask = image_dict["Masks"]["SpotMask"]["spotMask"]
        imsave = Image.fromarray(outlier_mask)
        imsave.save(
            os.path.join(
                self.directory,
                "masks",
                self.name + "-" + self.number + "_om.tif"
            )
        )
        single_iter_times.append(time.time())
        time_checkpoints.append("Outlier Mask")
        # close holes
        if self.logging:
            print("Closing the mask")
        if self.closing_method == "binary_closing":
            closed_mask = ski.morphology.binary_closing(
                outlier_mask, footprint=ski.morphology.square(3)
            )
            imsave = Image.fromarray(closed_mask)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_closedmask.tif"
                )
            )

        elif self.closing_method == "remove_small":
            closed_mask = ski.morphology.remove_small_holes(outlier_mask, 6)
            imsave = Image.fromarray(closed_mask)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_closedmask.tif"
                )
            )
        elif (self.closing_method == None) or (self.closing_method == ""):
            closed_mask = outlier_mask
        else:
            print("Unrecognized closing method: Using none")
            closed_mask = outlier_mask
        single_iter_times.append(time.time())
        time_checkpoints.append("Mask Closing")

        # split mask
        if self.logging:
            print("Splitting the mask")
        # spots, arcs = split_h_and_grad(image_dict['image'],image_dict['Image Controls']['center'],closed_mask,image_dict['grad'])
        # split_spots, split_arcs = current_splitting_method(image_dict['image'],closed_mask,image_dict['pixelQmap'],image_dict['gradient'])
        return_steps = False
        if return_steps:
            (
                split_spots,
                split_arcs,
                spots_table,
                base_arc,
                qgrad_arc,
                azim_grad_2,
                radial_grad_2,
            ) = current_splitting_method(
                image_dict["image"],
                closed_mask,
                self.cache["pixelQmap"],
                self.cache["pixelAzmap"],
                self.cache["gradient"],
                return_steps=return_steps,
                interpolate=False,
                predef_mask=nonpositive_mask,
                predef_mask_extended=predef_mask_extended,
            )
            imsave = Image.fromarray(split_spots)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_spots.tif"
                )
            )
            imsave = Image.fromarray(split_arcs)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_arcs.tif"
                )
            )
            imsave = Image.fromarray(base_arc)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_qwidth_arc.tif"
                )
            )
            imsave = Image.fromarray(qgrad_arc)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_qgrad_arc.tif"
                )
            )
            imsave = Image.fromarray(azim_grad_2)
            imsave.save(
                os.path.join(
                    self.directory,
                    "grads",
                    self.name + "-" + self.number + "_azim_grad_2.tif"
                )
            )
            imsave = Image.fromarray(radial_grad_2)
            imsave.save(
                os.path.join(
                    self.directory,
                    "grads",
                    self.name + "-" + self.number + "_radial_grad_2.tif"
                )
            )
        else:
            (
                split_spots,
                split_arcs,
                spots_table,
            ) = current_splitting_method(
                image_dict["image"],
                closed_mask,
                self.cache["pixelQmap"],
                self.cache["pixelAzmap"],
                self.cache["gradient"],
                return_steps=return_steps,
                interpolate=False,
                predef_mask=nonpositive_mask,
                predef_mask_extended=predef_mask_extended
            )
            imsave = Image.fromarray(split_spots)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_spots.tif"
                )
            )
            imsave = Image.fromarray(split_arcs)
            imsave.save(
                os.path.join(
                    self.directory,
                    "masks",
                    self.name + "-" + self.number + "_arcs.tif"
                )
            )
        single_iter_times.append(time.time())
        time_checkpoints.append("Mask Splitting 0.1")

        # alt_split_spots, alt_split_arcs, azim_grad_2 = alt_test(image_dict['image'],closed_mask,image_dict['gradient'])
        # imsave = Image.fromarray(alt_split_spots)
        # imsave.save(directory + '\\masks\\' + name + '-' + number + '_gradspots0p1.tif')
        # imsave = Image.fromarray(alt_split_arcs)
        # imsave.save(directory + '\\masks\\' + name + '-' + number + '_gradarcs0p1.tif')
        # single_iter_times.append(time.time())
        # time_checkpoints.append('No Q Mask Splitting 0.1')
        # imsave = Image.fromarray(azim_grad_2)
        # imsave.save(directory + '\\grads\\' + name + '-' + number + '_azimgrad2.tif')

        # integrate
        # Use unpolarized image when calling GSAS-II scriptable, as it polarizes the image by default
        # x_base,y_base,w_base = integrate(image,~mask_nonzero,cache['Image Controls'],cache['Masks'],im_mask,TA_blocks)
        # Outlier mask needs to be set in img data when calling each one with G2sc
        if self.logging:
            print("Integrating...")
        # img.data['Masks']['SpotMask']['spotMask'] = ~mask_nonzero
        # hist_base = img.Integrate(name = name + '-' + number,MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = ~mask_nonzero
        # hist_base = Integrate(img_copy,blkSize=self.blkSize,name = name + '-' + number,MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        hist_base = pytorch_integrate(
            image_dict["image"],
            frame_and_predef,
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        )
        # img.data['Masks']['SpotMask']['spotMask'] = outlier_mask
        # hist_om = img.Integrate(name=name + '-' + number + '_om',MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = outlier_mask
        # hist_om = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_om',MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        # img.data['Masks']['SpotMask']['spotMask'] = split_spots
        # hist_spotsmasked = img.Integrate(name=name + '-' + number + '_spots', MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = split_spots
        # hist_spotsmasked = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_spots', MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        # img.data['Masks']['SpotMask']['spotMask'] = split_arcs
        # hist_arcsmasked = img.Integrate(name=name + '-' + number + '_arcs', MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = split_arcs
        # hist_arcsmasked = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_arcs', MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        # img.data['Masks']['SpotMask']['spotMask'] = closed_mask
        # hist_closed = img.Integrate(name=name + '-' + number + '_closed', MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = closed_mask
        # hist_closed = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_closed', MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        hist_closed = pytorch_integrate(
            image_dict["image"],
            np.logical_or(closed_mask, frame_and_predef),
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        )
        # img.data['Masks']['SpotMask']['spotMask'] = split_spots_closed
        # hist_closedspotsmasked = img.Integrate(name=name + '-' + number + '_spotsclosed', MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = split_spots_closed
        # hist_closedspotsmasked = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_spotsclosed', MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        hist_closedspotsmasked = pytorch_integrate(
            image_dict["image"],
            np.logical_or(split_spots, frame_and_predef),
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        )
        # img.data['Masks']['SpotMask']['spotMask'] = split_arcs_closed
        # hist_closedarcsmasked = img.Integrate(name=name + '-' + number + '_arcsclosed', MaskMap=cache['intMaskMap'],ThetaAzimMap=cache['intTAmap'])
        # img_copy['Masks']['SpotMask']['spotMask'] = split_arcs_closed
        # hist_closedarcsmasked = Integrate(img_copy,blkSize=self.blkSize,name=name + '-' + number + '_arcsclosed', MaskMap=self.cache['intMaskMap'],ThetaAzimMap=self.cache['intTAmap'])
        hist_closedarcsmasked = pytorch_integrate(
            image_dict["image"],
            np.logical_or(split_arcs, frame_and_predef),
            self.cache["tth_idx"],
            self.cache["tth_val"],
            self.cache["raveled_pol"],
            self.cache["raveled_dist"],
            self.cache["tth_size"],
        )
        if self.logging:
            print("Integration complete")
        single_iter_times.append(time.time())
        time_checkpoints.append("Integration")
        # save integrals
        integral_file_base = os.path.join(
            self.directory,
            "integrals",
            self.name + "-" + self.number
        )
        # print(len(hist_base), len(hist_om), len(hist_spotsmasked), len(hist_arcsmasked), len(hist_closed), len(hist_closedspotsmasked), len(hist_closedarcsmasked))
        # hist_base[0].Export(directory + '\\integrals\\' + name + '-' + number + '_base.xye','.xye')
        # Export_xye(hist_base[0],directory + '\\integrals\\' + name + '-' + number + '_base.xye')
        Export_xye(
            self.name + "-" + self.number + "_base",
            hist_base.T,
            integral_file_base + "_base.xye",
            error=False,
        )
        # hist_om[0].Export(directory + '\\integrals\\' + name + '-' + number + '_om.xye','.xye')
        # Export_xye(hist_om[0],directory + '\\integrals\\' + name + '-' + number + '_om.xye')
        # hist_spotsmasked[0].Export(directory + '\\integrals\\' + name + '-' + number + '_spotsmasked.xye','.xye')
        # Export_xye(hist_spotsmasked[0],directory + '\\integrals\\' + name + '-' + number + '_spotsmasked.xye')
        # hist_arcsmasked[0].Export(directory + '\\integrals\\' + name + '-' + number + '_arcsmasked.xye','.xye')
        # Export_xye(hist_arcsmasked[0],directory + '\\integrals\\' + name + '-' + number + '_arcsmasked.xye')
        # hist_closed[0].Export(directory + '\\integrals\\' + name + '-' + number + '_closed.xye','.xye')
        # Export_xye(hist_closed[0],directory + '\\integrals\\' + name + '-' + number + '_closed.xye')
        Export_xye(
            self.name + "-" + self.number + "_closed",
            hist_closed.T,
            integral_file_base + "_closed.xye",
            error=False,
        )
        # hist_closedspotsmasked[0].Export(directory + '\\integrals\\' + name + '-' + number + '_closedspotsmasked.xye','.xye')
        # Export_xye(hist_closedspotsmasked[0],directory + '\\integrals\\' + name + '-' + number + '_closedspotsmasked.xye')
        Export_xye(
            self.name + "-" + self.number + "_closedspotsmasked",
            hist_closedspotsmasked.T,
            integral_file_base + "_closedspotsmasked.xye",
            error=False,
        )
        # hist_closedarcsmasked[0].Export(directory + '\\integrals\\' + name + '-' + number + '_closedarcsmasked.xye','.xye')
        # Export_xye(hist_closedarcsmasked[0],directory + '\\integrals\\' + name + '-' + number + '_closedarcsmasked.xye')
        Export_xye(
            self.name + "-" + self.number + "_closedarcsmasked",
            hist_closedarcsmasked.T,
            integral_file_base + "_closedarcsmasked.xye",
            error=False,
        )
        single_iter_times.append(time.time())
        time_checkpoints.append("Writing integrals to disk")
        # delete temporary project
        # if logging: print("Deleting project")
        # img.clearImageCache()
        # img.clearPixelMask()
        # del gpx
        # single_iter_times.append(time.time())

        # stats
        stats_prefix = os.path.join(self.directory, "stats", self.name)
        # spots stats
        spots_table.to_csv(stats_prefix + "-" + self.number + "_spots_stats.csv")
        # ~ 950 KB for table
        # 2d histogram: area, Q position
        # ~7800 KB for 1000x1000 bin histogram
        # 81 KB for 100x100 bin histogram
        hist, x_edges, y_edges = np.histogram2d(
            spots_table["area"].values, spots_table["intensity_mean"].values, 100
        )
        with open(
            stats_prefix + "-" + self.number + "_spots_hist.npy", "wb"
        ) as outfile:
            np.save(outfile, hist)
            np.save(outfile, x_edges)
            np.save(outfile, y_edges)

        # Calculate comparisons between images
        # Find and read in previous image given current image number
        prev_number = ""
        number_int_prev = int(self.number) - 1
        if number_int_prev < 0:
            # first image (00000) will have no previous image; just compare to self
            prev_number = self.number
        else:
            # turn int back to '00001' format, padded to 5 digits
            prev_number = f"{number_int_prev:05}"
        try:
            previous_image = ski.io.imread(
                os.path.join(self.directory, self.name + "-" + prev_number + self.ext)
            ).astype(np.float32)
        except:
            previous_image = image_dict["image"].astype(np.float32)
        csim_f = 1 - spatial.distance.cosine(
            np.array(image_dict["image"], dtype=np.float32).ravel(),
            self.cache["First image"].ravel(),
        )
        csim_p = 1 - spatial.distance.cosine(
            np.array(image_dict["image"], dtype=np.float32).ravel(),
            previous_image.ravel(),
        )
        # self.csim_first.append(csim_f)
        # self.csim_prev.append(csim_p)
        # print("csim: time {0:.2f}s".format((time.time() - t)/2))
        # Not safe for multiprocessing
        # if not os.path.exists(stats_prefix + '_csim.txt'):
        #     with open(stats_prefix + '_csim.txt','w') as outfile:
        #         outfile.write("Cosine similarity as compared to:\n")
        #         outfile.write("First image\tPrevious image\n")
        #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=csim_f,prev=csim_p))
        # else:
        #     with open(stats_prefix + '_csim.txt','a') as outfile:
        #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=csim_f,prev=csim_p))
        with open(stats_prefix + "-" + self.number + "_csim.txt", "w") as outfile:
            outfile.write(
                "{first:0.4f}\t{prev:0.4f}\n".format(first=csim_f, prev=csim_p)
            )

        # single_iter_times.append(time.time())
        # t = time.time()
        # # nmi_f = ski.metrics.normalized_mutual_information(image_dict['image'],self.cache['First image'])
        # # nmi_p = ski.metrics.normalized_mutual_information(image_dict['image'],self.cache['Previous image'])
        # # self.nmi_first.append(nmi_f)
        # # self.nmi_prev.append(nmi_p)
        # print("nmi: {0:.2f}s".format((time.time() - t)/2))
        # # if not os.path.exists(stats_prefix + '_nmi.txt'):
        # #     with open(stats_prefix + '_nmi.txt','w') as outfile:
        # #         outfile.write("Normalized Mutual Information as compared to:\n")
        # #         outfile.write("First image\tPrevious image\n")
        # #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=nmi_f,prev=nmi_p))
        # # else:
        # #     with open(stats_prefix + '_nmi.txt','a') as outfile:
        # #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=nmi_f,prev=nmi_p))
        # single_iter_times.append(time.time())
        # t = time.time()
        # # data_min = min(np.min(image_dict['image']),np.min(self.cache['First image']))
        # # data_max = max(np.max(image_dict['image']),np.max(self.cache['First image']))
        # # ssim_f = ski.metrics.structural_similarity(image_dict['image'],self.cache['First image'],data_range=data_max-data_min)
        # # self.ssim_first.append(ssim_f)
        # # data_min = min(np.min(image_dict['image']),np.min(self.cache['Previous image']))
        # # data_max = max(np.max(image_dict['image']),np.max(self.cache['Previous image']))
        # # ssim_p = ski.metrics.structural_similarity(image_dict['image'],self.cache['Previous image'],data_range=data_max-data_min)
        # # self.ssim_prev.append(ssim_p)
        # print("ssim: {0:.2f}".format((time.time() - t)/2))
        # # if not os.path.exists(stats_prefix + '_ssim.txt'):
        # #     with open(stats_prefix + '_ssim.txt','w') as outfile:
        # #         outfile.write("Structural similarity as compared to:\n")
        # #         outfile.write("First image\tPrevious image\n")
        # #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=ssim_f,prev=ssim_p))
        # # else:
        # #     with open(stats_prefix + '_ssim.txt','a') as outfile:
        # #         outfile.write("{first:0.4f}\t{prev:0.4f}\n".format(first=ssim_f,prev=ssim_p))

        single_iter_times.append(time.time())
        time_checkpoints.append("Blank")
        # Also not safe for multiprocessing. Need to grab number-1 when it exists
        self.cache["Previous image"] = image_dict["image"]

        for i in range(len(single_iter_times) - 1):
            print(
                "{0}: {1:.2f}".format(
                    time_checkpoints[i], single_iter_times[i + 1] - single_iter_times[i]
                )
            )
        # self.all_times.append(single_iter_times[-1]-single_iter_times[0])

        self.finished.emit()


class main_window(QtWidgets.QWidget):
    # text box/file browser directory location
    # ditto config file
    # ditto predef mask
    # start button
    # clear queue button
    # optional "choose existing files to run over" section
    # default: none, shortcut button for all, else choose which files
    def __init__(self, directory=None, imctrl=None, imgmask=None):
        super().__init__()
        # self.directory_text = QtWidgets.QPushButton("Directory:")
        # self.directory_loc = QtWidgets.QLabel()
        self.directory_widget = file_select(
            "Directory:",
            default_text=directory,
            isdir=True,
        )
        # self.config_text = QtWidgets.QPushButton("Config file:")
        # self.config_loc = QtWidgets.QLabel()
        self.config_widget = file_select(
            "Config file:",
            default_text=imctrl,
            startdir=self.directory_widget.file_name.text(),
            ext="Imctrl and PONI files (*.imctrl *.poni)",
        )
        # self.predef_mask_text = QtWidgets.QPushButton("Predefined Mask:")
        # self.predef_mask_loc = QtWidgets.QLabel()
        self.flatfield_widget = file_select(
            "Flat-field file:", startdir=self.directory_widget.file_name.text()
        )
        self.predef_mask_widget = file_select(
            "Predefined Mask:",
            default_text=imgmask,
            startdir=self.directory_widget.file_name.text(),
        )
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.released.connect(self.start_button_pressed)
        self.clear_queue_button = QtWidgets.QPushButton("Clear Queue")
        self.clear_queue_button.released.connect(self.clear_queue_pressed)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.released.connect(self.stop_button_pressed)
        self.stop_button.setEnabled(False)
        # tooltip for clear queue button which notes that program will finish processing current item
        # exit button with "are you sure you're done processing everything" pop-up
        self.process_existing_images_checkbox = QtWidgets.QCheckBox(
            "Process existing images"
        )
        self.regex_label = QtWidgets.QLabel("Regex for existing images:")
        self.existing_images_regex = QtWidgets.QTextEdit()

        # self.time_checkpoints = ["Start","Image loaded","Cache","Zero mask","Polar-correct","Outlier mask","Closing mask","Split first mask","Split second mask","All integrations","Save integrals","Delete project"]
        # self.time_checkpoints = ["Start", "Cache", "Zero mask", "Outlier mask", "Closing mask", "Splitting mask", "Integrations", "Save integrals", "CSim", "NMI", "SSim"]
        self.all_times = []
        # set up gsas-ii project
        # G2sc.blkSize = 2**8  # computer-dependent tuning parameter
        self.blkSize = 2**8
        # G2sc.SetPrintLevel('warn')   # reduces output

        self.cache = {}  # place to save intermediate computations

        self.queue = deque()
        self.event_handler = image_monitor(self.queue)
        self.stop_event = threading.Event()
        # self.watchdog_thread = threading.Thread(target=watchdog_observer,args=(self.directory,self.event_handler),daemon=True)

        self.timer = QtCore.QTimer()
        self.keep_running = False
        self.timer.timeout.connect(self.on_timeout)

        # self.iteration_thread = QtCore.QThread()
        # self.cache_thread = QtCore.QThread()

        self.queue_length_info = QtWidgets.QLabel(
            "Queue is {0} items long".format(len(self.queue))
        )

        # self.is_running_process = False

        self.window_layout = QtWidgets.QGridLayout()
        self.window_layout.addWidget(self.directory_widget, 0, 0, 1, 3)
        self.window_layout.addWidget(self.config_widget, 1, 0, 1, 3)
        self.window_layout.addWidget(self.flatfield_widget, 2, 0, 1, 3)
        self.window_layout.addWidget(self.predef_mask_widget, 3, 0, 1, 3)
        self.window_layout.addWidget(self.start_button, 4, 0)
        self.window_layout.addWidget(self.clear_queue_button, 4, 1)
        self.window_layout.addWidget(self.stop_button, 4, 2)
        self.window_layout.addWidget(self.process_existing_images_checkbox, 5, 0)
        self.window_layout.addWidget(self.queue_length_info, 6, 0)
        # self.window_layout.addWidget(self.regex_label,7,0)
        # self.window_layout.addWidget(self.existing_images_regex,8,0)

        self.setLayout(self.window_layout)
        self.show()

    def cache_thread_finished(self):
        self.has_made_cache = True
        self.cache_thread = None
        if self.keep_running:
            self.timer.start()

    def iteration_thread_finished(self):
        if self.keep_running:
            self.timer.start()

    def on_timeout(self):
        if self.keep_running:
            # block=True in Queue.get() tells it to wait until there is something in the queue to grab it
            # can also set a timeout value (in seconds) to wait before it throws an Empty exception
            # Windows systems apparently have a problem with block=True, timeout=None
            # filename,name,number = queue.get(block=True,timeout=30)
            if self.queue:
                self.queue_length_info.setText(
                    "Queue is {0} items long".format(len(self.queue))
                )
                if self.has_made_cache:
                    # ensure it's been some time since the file was modified
                    if time.time() - os.path.getmtime(self.queue[0][0]) > 1:
                        filename, name, number, ext = self.queue.popleft()
                        print(filename, name, number, ext)
                        # print("Queue is {0} items long".format(len(self.queue)))
                        # self.single_iteration(filename,self.imgctrl,self.imgmask,self.directory,name,number)
                        # set up iteration thread. Should set these up with a pool and just run, but for now, run one at a time.
                        self.timer.stop()
                        self.iteration_thread = QtCore.QThread()
                        self.iteration_worker = SingleIterator(
                            self.cache,
                            filename,
                            self.imgctrl,
                            self.imgmask,
                            self.directory,
                            name,
                            number,
                            ext,
                        )
                        self.iteration_worker.moveToThread(self.iteration_thread)
                        self.iteration_thread.started.connect(self.iteration_worker.run)
                        self.iteration_worker.finished.connect(
                            self.iteration_thread.quit
                        )
                        self.iteration_worker.finished.connect(
                            self.iteration_worker.deleteLater
                        )
                        # self.iteration_thread.finished.connect(self.iteration_thread.deleteLater)
                        self.iteration_thread.finished.connect(
                            self.iteration_thread_finished
                        )
                        self.iteration_thread.start()
                        # print("Queue is {0} items long".format(len(self.queue)))
                        # self.queue_length_info.setText("Queue is {0} items long".format(len(self.queue)))
                        # test_iteration(filename)
                        # wait_start = time.time()
                else:
                    # check that it's been a moment
                    if time.time() - os.path.getmtime(self.queue[0][0]) > 1:
                        # set up cache thread
                        self.timer.stop()
                        self.cache_thread = QtCore.QThread()
                        filename = self.queue[0][0]
                        print(filename)
                        self.cache_worker = CacheCreator(
                            self.cache,
                            self.directory,
                            filename,
                            self.imgctrl,
                            self.flatfield,
                            self.imgmask,
                            self.blkSize,
                        )
                        self.cache_worker.moveToThread(self.cache_thread)
                        self.cache_thread.started.connect(self.cache_worker.run)
                        self.cache_worker.finished.connect(self.cache_thread.quit)
                        self.cache_worker.finished.connect(
                            self.cache_worker.deleteLater
                        )
                        # self.cache_thread.finished.connect(self.cache_thread.deleteLater)
                        self.cache_thread.finished.connect(self.cache_thread_finished)
                        self.cache_thread.start()
            else:
                self.queue_length_info.setText("Queue is 0 items long")
            # else:
            #    #If it's been over an hour since the last update, stop
            #    if time.time() - wait_start > 60:
            #        print("Average time: {0}".format(np.average(self.all_times)))
            #        self.keep_running = False
        else:
            self.timer.stop()

    def start_processing(self):
        self.directory = self.directory_widget.file_name.text()
        self.imgctrl = self.config_widget.file_name.text()
        self.imgmask = self.predef_mask_widget.file_name.text()
        self.flatfield = self.flatfield_widget.file_name.text()
        self.cache = {}
        self.has_made_cache = False
        # print("Directory: {0}, Ctrl file: {1}, Predef mask: {2}".format(dir_name,ctrl_name,predef_mask))
        # self.process = main_process(dir_name,ctrl_name,predef_mask)
        # create subdirectories if needed
        newdirs = ["maps", "masks", "integrals", "stats", "grads"]
        for newdir in newdirs:
            path = os.path.join(self.directory, newdir)  # store maps with the images
            if not os.path.exists(path):
                os.mkdir(path)

        # Grab existing file names and add them to the queue if option checked
        if self.process_existing_images_checkbox.isChecked():
            # existing_files = glob.glob(self.directory+"/*.metadata")
            existing_files = glob.glob(self.directory + "/*.tif")
            # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif.metadata$"
            # reg_tif = r"(?P<directory>.*\\)(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})\.tif$"
            reg_image = r"(?P<directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
            # number -> actual int
            # number_int = results[0].group("number").remove("-").remove("_")
            # number_int = int(number_int)
            for filename in existing_files:
                # Add file path to queue, stripping ".metadata"
                results = re.match(reg_image, filename)
                # Regex observer uses re.findall(), so it needs results[0].
                # self.queue.append([filename[:-9],results.group("name"),results.group("number")])
                if results is not None:
                    self.queue.append(
                        [
                            filename,
                            results.group("name"),
                            results.group("number"),
                            results.group("ext"),
                        ]
                    )
                    print(
                        filename,
                        results.group("name"),
                        results.group("number"),
                        results.group("ext"),
                    )

        # Start queue
        print("Starting queue")

        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.directory, recursive=False)
        self.observer.start()

        # main function to cycle, calls iteration while there are new images to process
        self.keep_running = True
        self.csim_first = []
        self.csim_prev = []
        self.nmi_first = []
        self.nmi_prev = []
        self.ssim_first = []
        self.ssim_prev = []
        self.timer.start(100)

    def clear_queue(self):
        self.queue.clear()
        self.queue_length_info.setText(
            "Queue is {0} items long".format(len(self.queue))
        )

    def pause(self):
        self.keep_running = False
        # watchdog thread will still keep populating the queue

    def resume(self):
        self.keep_running = True
        self.timer.start(100)

    def update_dir(self, directory):
        self.pause()
        self.clear_queue()
        self.directory = directory
        # self.watchdog_thread = threading.Thread(target=watchdog_observer,args=(self.directory,self.event_handler),daemon=True)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.directory, recursive=False)
        self.observer.start()
        # self.resume()

    def start_button_pressed(self):
        if self.start_button.text() == "Start":
            self.start_processing()
            self.start_button.setText("Pause")
            self.stop_button.setEnabled(True)
            self.directory_widget.setEnabled(False)
            self.config_widget.setEnabled(False)
            self.flatfield_widget.setEnabled(False)
            self.predef_mask_widget.setEnabled(False)
        elif self.start_button.text() == "Pause":
            self.pause()
            self.start_button.setText("Resume")
        elif self.start_button.text() == "Resume":
            self.resume()
            self.start_button.setText("Pause")

    def clear_queue_pressed(self):
        print("Clearing queue")
        self.clear_queue()

    def stop_button_pressed(self):
        print("Stopping and clearing queue")
        self.stop_button.setText("Stopping...")
        # disable all
        self.start_button.setEnabled(False)
        self.clear_queue_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause()
        self.clear_queue()
        # self.watchdog_thread.stop()
        # self.watchdog_thread.join()
        self.observer.stop()
        self.observer.join()
        # if self.cache_thread.isRunning():
        if not self.has_made_cache:
            self.cache_thread.quit()
            self.cache_thread.finished.connect(self.really_stopped)
            self.cache_thread.finished.connect(self.cache_thread.deleteLater)
        elif self.iteration_thread.isRunning():
            self.iteration_thread.quit()
            self.iteration_thread.finished.connect(self.really_stopped)
            self.iteration_thread.finished.connect(self.iteration_thread.deleteLater)
        else:
            # self.is_running_process = False
            self.start_button.setText("Start")
            self.start_button.setEnabled(True)
            self.clear_queue_button.setEnabled(True)
            self.stop_button.setText("Stop")
            self.directory_widget.setEnabled(True)
            self.config_widget.setEnabled(True)
            self.flatfield_widget.setEnabled(True)
            self.predef_mask_widget.setEnabled(True)

    def really_stopped(self):
        # self.is_running_process = False
        self.start_button.setText("Start")
        self.start_button.setEnabled(True)
        self.clear_queue_button.setEnabled(True)
        self.stop_button.setText("Stop")
        self.directory_widget.setEnabled(True)
        self.config_widget.setEnabled(True)
        self.flatfield_widget.setEnabled(True)
        self.predef_mask_widget.setEnabled(True)

    def closeEvent(self, evt):
        # if not self.is_running_process:
        #    evt.accept()
        if not self.stop_button.isEnabled():
            # button disabled while not actively running a process; no need to prompt
            evt.accept()
        elif (
            QtWidgets.QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to stop running over all data?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            == QtWidgets.QMessageBox.StandardButton.Yes
        ):
            self.clear_queue()
            self.pause()
            evt.accept()
        else:
            evt.ignore()


# Following https://gsas-ii.readthedocs.io/en/latest/GSASIIscriptable.html 16.7.9. Optimized Image Integration
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory")
    parser.add_argument("-c", "--imctrl")
    parser.add_argument("-m", "--imgmask", default=None)
    args = parser.parse_args()

    # Pass in location and names of files
    dataLoc = os.path.abspath(
        os.path.split(__file__)[0]
    )  # data in location of this file
    PathWrap = lambda fil: os.path.join(
        dataLoc, fil
    )  # convenience function for file paths

    if args.imgmask is not None:
        imgmask = PathWrap(args.imgmask)
    else:
        imgmask = None
    if args.directory:
        directory = PathWrap(args.directory)
    else:
        directory = None
    if args.imctrl:
        if os.path.exists(PathWrap(args.imctrl)):
            imgctrl = PathWrap(args.imctrl)
        elif os.path.exists(os.path.join(directory, args.imctrl)):
            imgctrl = os.path.join(directory, args.imctrl)
        else:
            print(
                "Image control file not found in this directory or in specified directory."
            )
            imgctrl = None
    else:
        imgctrl = None

    app = QtWidgets.QApplication([])
    window = main_window(directory=directory, imctrl=imgctrl, imgmask=imgmask)
    sys.exit(app.exec())
