"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file lists imports from GSASII used in the analysis pipeline.
The GSASII license can be found in the GSASII_LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/GSASII_LICENSE
"""

import os, sys
import numpy as np
import skimage as ski
import tifffile as tf
import time

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
bindist_dir = os.path.join(script_dir, "bindist")
# Add 'bindist' to the beginning of sys.path
print(bindist_dir)
sys.path.insert(0, bindist_dir)

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

def sind(x):
    return npsind(x)

def asind(x):
    return npasind(x)

def cosd(x):
    return npcosd(x)

def acosd(x):
    return npacosd(x)

def tand(x):
    return nptand(x)

def atand(x):
    return npatand(x)

def atan2d(y, x):
    return npatan2d(y, x)


# load data from file
def load_image_tifffile(imloc):
    image = tf.imread(imloc)
    return image


def load_image(imloc):
    image = ski.io.imread(imloc)
    return image


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


# recreate for poni files
def LoadControlsPONI(Slines, data):
    "Read values from a .poni image control file."
    cntlList = [
        "Detector_config",
        "Distance",
        "Poni1",
        "Poni2",
        "Rot1",
        "Rot2",
        "Rot3",
        "Wavelength"
    ]
    initial_data = {}
    for S in Slines:
        if S[0] == "#":
            continue
        [key, val] = S.strip().split(":", 1)
        if key in [
            "Detector_config"
        ]:
            initial_data[key] = eval(val) # dictionary
        elif key in [
            "Distance",
            "Poni1",
            "Poni2",
            "Rot1",
            "Rot2",
            "Rot3",
            "Wavelength"
        ]:
            initial_data[key] = float(val)
    # now convert
    save = poni_to_gsasii(initial_data)
    # add info for number of outchannels, iotth
    # save["outChannels"] = 2500 # seems a default in the imctrl files
    # save["IOtth"] = [0.8, 16.8] # needs to be adjusted/determined
    # save["PolaVal"] = [0.9, False] # boolean unclear; should probably set to 1.0 and tell people to provide polarization correction file or number if needed
    # temporarily overwrite with what they should be in case this is incorrectly converting:
    # save["distance"] = 689.5465510332797
    # save["tilt"] = -0.16803895763020923
    # save["rotation"] = 327.8798449443938
    # save["center"] = [214.78748452404744, 216.55137107349705]
    # save["wavelength"] = 0.24087

    data.update(save)

# from pyfai.geometry.fit2d.convert_to_Fit2d
def poni_to_gsasii(controls):
    cos_tilt = np.cos(controls["Rot1"]) * np.cos(controls["Rot2"])
    sin_tilt = np.sqrt(1.0 - cos_tilt * cos_tilt)
    tan_tilt = sin_tilt / cos_tilt

    # PyFai's 'tilt plane rotation'
    if sin_tilt == 0:
        # tilt plane rotation undefined when there is no tilt
        cos_tilt = 1.0
        sin_tilt = 0.0
        cos_tpr = 1.0
        sin_tpr = 0.0
    else:
        cos_tpr = max(-1.0, min(1.0, -np.cos(controls["Rot2"]) * np.sin(controls["Rot1"]) / sin_tilt))
        sin_tpr = np.sin(controls["Rot2"]) / sin_tilt
    directDist = 1.0e3 * controls["Distance"] / cos_tilt
    tilt = np.degrees(np.arccos(cos_tilt))
    if sin_tpr < 0:
        tpr = -np.degrees(np.arccos(cos_tpr))
    else:
        tpr = np.degrees(np.arccos(cos_tpr))

    # centerX = (controls["Poni2"] + controls["Distance"] * tan_tilt * cos_tpr) / controls["Detector_config"]["pixel2"]
    centerX = (controls["Poni2"] + controls["Distance"] * tan_tilt * cos_tpr) * 1000
    if abs(tilt) < 1e-5: # in degrees
        # centerY = (controls["Poni1"]) / controls["Detector_config"]["pixel1"]
        centerY = (controls["Poni1"]) * 1000
    else:
        # centerY = (controls["Poni1"] + controls["Distance"] * tan_tilt * sin_tpr) / controls["Detector_config"]["pixel1"]
        centerY = (controls["Poni1"] + controls["Distance"] * tan_tilt * sin_tpr) * 1000
    
    # further corrections to match GSASII: angle is off by 90 degrees
    
    tilt = -tilt
    tpr = 360-tpr
    if tpr > 360:
        tpr -= 360
    tpr -= 90.
    if tpr < 0:
        tpr += 360

    out = {}
    out["distance"] = directDist
    out["tilt"] = tilt
    out["rotation"] = tpr
    out["center"] = [centerX, centerY]
    out["wavelength"] = controls["Wavelength"] * 1e10
    print(out)

    return out


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

# G2img_1TIF.py
def GetTifData(filename, DEBUG=False):
    '''Read an image in a pseudo-tif format,
    as produced by a wide variety of software, almost always
    incorrectly in some way. 
    '''
    import struct as st
    import array as ar
    image = None
    File = open(filename,'rb')
    dataType = 5
    center = [None,None]
    wavelength = None
    distance = None
    polarization = None
    samplechangerpos = None
    xpixelsize = None
    ypixelsize = None
    pixy = None
    try:
        Meta = open(filename+'.metadata','r')
        head = Meta.readlines()
        for line in head:
            line = line.strip()
            try:
                if '=' not in line: continue
                keyword = line.split('=')[0].strip()
                if 'dataType' == keyword:
                    dataType = int(line.split('=')[1])
                elif 'wavelength' == keyword.lower():
                    wavelength = float(line.split('=')[1])
                elif 'distance' == keyword.lower():
                    distance = float(line.split('=')[1])
                elif 'polarization' == keyword.lower():
                    polarization = float(line.split('=')[1])
                elif 'samplechangercoordinate' == keyword.lower():
                    samplechangerpos = float(line.split('=')[1])
                elif 'detectorxpixelsize' == keyword.lower():
                    xpixelsize = float(line.split('=')[1])
                elif 'detectorypixelsize' == keyword.lower():
                    ypixelsize = float(line.split('=')[1])
            except:
                print('error reading metadata: '+line)
        Meta.close()
    except IOError:
        if DEBUG:
            print ('no metadata file found - will try to read file anyway')
        head = ['no metadata file found',]
        
    tag = File.read(2)
    if 'bytes' in str(type(tag)):
        tag = tag.decode('latin-1')
    byteOrd = '<'
    if tag == 'II' and int(st.unpack('<h',File.read(2))[0]) == 42:     #little endian
        IFD = int(st.unpack(byteOrd+'i',File.read(4))[0])
    elif tag == 'MM' and int(st.unpack('>h',File.read(2))[0]) == 42:   #big endian
        byteOrd = '>'
        IFD = int(st.unpack(byteOrd+'i',File.read(4))[0])        
    else:
        lines = ['not a detector tiff file',]
        return lines,0,0,0
    File.seek(IFD)                                                  #get number of directory entries
    NED = int(st.unpack(byteOrd+'h',File.read(2))[0])
    IFD = {}
    nSlice = 1
    if DEBUG: print('byteorder:',byteOrd)
    for ied in range(NED):
        Tag,Type = st.unpack(byteOrd+'Hh',File.read(4))
        nVal = st.unpack(byteOrd+'i',File.read(4))[0]
        if DEBUG: print ('Try:',Tag,Type,nVal)
        if Type == 1:
            Value = st.unpack(byteOrd+nVal*'b',File.read(nVal))
        elif Type == 2:
            Value = st.unpack(byteOrd+'i',File.read(4))
        elif Type == 3:
            Value = st.unpack(byteOrd+nVal*'h',File.read(nVal*2))
            st.unpack(byteOrd+nVal*'h',File.read(nVal*2))
        elif Type == 4:
            if Tag in [273,279]:
                nSlice = nVal
                nVal = 1
            Value = st.unpack(byteOrd+nVal*'i',File.read(nVal*4))
        elif Type == 5:
            Value = st.unpack(byteOrd+nVal*'i',File.read(nVal*4))
        elif Type == 11:
            Value = st.unpack(byteOrd+nVal*'f',File.read(nVal*4))
        IFD[Tag] = [Type,nVal,Value]
        if DEBUG: print (Tag,IFD[Tag])
    sizexy = [IFD[256][2][0],IFD[257][2][0]]
    [nx,ny] = sizexy
    Npix = nx*ny
    time0 = time.time()
    if 34710 in IFD:
        print ('Read MAR CCD tiff file: '+filename)
        import ReadMarCCDFrame as rmf
        marFrame = rmf.marFrame(File,byteOrd,IFD)
        image = np.array(np.asarray(marFrame.image),dtype=np.int32)
        image = np.reshape(image,sizexy)
        if marFrame.origin:     #not upper left?
            image = np.flipud(image)
        if marFrame.viewDirection:  #view through sample to detector instead of TOWARD_SOURCE
            image = np.fliplr(image)
        tifType = marFrame.filetitle
        pixy = [marFrame.pixelsizeX/1000.0,marFrame.pixelsizeY/1000.0]
        head = marFrame.outputHead()
# extract resonable wavelength from header
        wavelength = marFrame.sourceWavelength*1e-5
        wavelength = (marFrame.opticsWavelength > 0) and marFrame.opticsWavelength*1e-5 or wavelength
        wavelength = (wavelength <= 0) and None or wavelength
# extract resonable distance from header
        distance = (marFrame.startXtalToDetector+marFrame.endXtalToDetector)*5e-4
        distance = (distance <= marFrame.startXtalToDetector*5e-4) and marFrame.xtalToDetector*1e-3 or distance
        distance = (distance <= 0) and None or distance
# extract resonable center from header
        center = [marFrame.beamX*marFrame.pixelsizeX*1e-9,marFrame.beamY*marFrame.pixelsizeY*1e-9]
        center = (center[0] != 0 and center[1] != 0) and center or [None,None]
#print head,tifType,pixy
    elif nSlice > 1:    #CheMin multislice tif file!
        try:
            import Image as Im
        except ImportError:
            try:
                from PIL import Image as Im
            except ImportError:
                print ("PIL/pillow Image module not present. This TIF cannot be read without this")
                #raise Exception("PIL/pillow Image module not found")
                lines = ['not a detector tiff file',]
                return lines,0,0,0
        tifType = 'CheMin'
        pixy = [40.,40.]
        image = np.flipud(np.array(Im.open(filename)))*10.
        distance = 18.0
        center = [pixy[0]*sizexy[0]/2000,0]     #the CheMin beam stop is here
        wavelength = 1.78892
    elif 272 in IFD:
        ifd = IFD[272]
        File.seek(ifd[2][0])
        S = File.read(ifd[1])
        if b'PILATUS' in S:
            tifType = 'Pilatus'
            dataType = 0
            pixy = [172.,172.]
            File.seek(4096)
            print ('Read Pilatus tiff file: '+filename)
            image = np.array(np.frombuffer(File.read(4*Npix),dtype=np.int32),dtype=np.int32)
        else:
            if IFD[258][2][0] == 16:
                if sizexy == [3888,3072] or sizexy == [3072,3888]:
                    tifType = 'Dexela'
                    pixy = [74.8,74.8]
                    print ('Read Dexela detector tiff file: '+filename)
                else:
                    tifType = 'GE'
                    pixy = [200.,200.]
                    print ('Read GE-detector tiff file: '+filename)
                File.seek(8)
                image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
            elif IFD[258][2][0] == 32:
                # includes CHESS & Pilatus files from Area Detector
                tifType = 'CHESS'
                pixy = [200.,200.]
                File.seek(8)
                print ('Read as 32-bit unsigned (CHESS) tiff file: '+filename)
                image = np.array(ar.array('I',File.read(4*Npix)),dtype=np.uint32)
    elif 270 in IFD:
        File.seek(IFD[270][2][0])
        S = File.read(IFD[273][2][0]-IFD[270][2][0])
        if b'Pilatus3' in S:
            tifType = 'Pilatus3'
            dataType = 0
            pixy = [172.,172.]
            File.seek(IFD[273][2][0])
            print ('Read Pilatus3 tiff file: '+filename)
            image = np.array(np.frombuffer(File.read(4*Npix),dtype=np.int32),dtype=np.int32)
        elif b'ImageJ' in S:
            tifType = 'ImageJ'
            dataType = 0
            pixy = [200.,200.]*IFD[277][2][0]
            File.seek(IFD[273][2][0])
            print ('Read ImageJ tiff file: '+filename)
            if IFD[258][2][0] == 32:
                image = File.read(4*Npix)
                image = np.array(np.frombuffer(image,dtype=byteOrd+'i4'),dtype=np.int32)
            elif IFD[258][2][0] == 16:
                image = File.read(2*Npix)
                pixy = [109.92,109.92]      #for LCLS ImageJ tif files
                image = np.array(np.frombuffer(image,dtype=byteOrd+'u2'),dtype=np.int32)
        else:   #gain map from  11-ID-C?
            pixy = [200.,200.]
            tifType = 'Gain map'
            image = File.read(4*Npix)
            image = np.array(np.frombuffer(image,dtype=byteOrd+'f4')*1000,dtype=np.int32)
            
    elif 262 in IFD and IFD[262][2][0] > 4:
        tifType = 'DND'
        pixy = [158.,158.]
        File.seek(512)
        print ('Read DND SAX/WAX-detector tiff file: '+filename)
        image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
    elif sizexy == [1536,1536]:
        tifType = 'APS Gold'
        pixy = [150.,150.]
        File.seek(64)
        print ('Read Gold tiff file:'+filename)
        image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
    elif sizexy == [2048,2048] or sizexy == [1024,1024] or sizexy == [3072,3072]:
        if IFD[273][2][0] == 8:
            if IFD[258][2][0] == 32:
                tifType = 'PE'
                pixy = [200.,200.]
                File.seek(8)
                print ('Read APS PE-detector tiff file: '+filename)
                if dataType == 5:
                    image = np.array(np.frombuffer(File.read(4*Npix),dtype=np.float32),dtype=np.int32)  #fastest
                else:
                    image = np.array(np.frombuffer(File.read(4*Npix),dtype=np.int32),dtype=np.int32)
            elif IFD[258][2][0] == 16: 
                tifType = 'MedOptics D1'
                pixy = [46.9,46.9]
                File.seek(8)
                print ('Read MedOptics D1 tiff file: '+filename)
                image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
                  
        elif IFD[273][2][0] == 4096:
            if sizexy[0] == 3072:
                pixy =  [73.,73.]
                tifType = 'MAR225'            
            else:
                pixy = [158.,158.]
                tifType = 'MAR325'            
            File.seek(4096)
            print ('Read MAR CCD tiff file: '+filename)
            image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
        elif IFD[273][2][0] == 512:
            tifType = '11-ID-C'
            pixy = [200.,200.]
            File.seek(512)
            print ('Read 11-ID-C tiff file: '+filename)
            image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
                    
    elif sizexy == [4096,4096]:
        if IFD[273][2][0] == 8:
            if IFD[258][2][0] == 16:
                tifType = 'scanCCD'
                pixy = [9.,9.]
                File.seek(8)
                print ('Read APS scanCCD tiff file: '+filename)
                image = np.array(ar.array('H',File.read(2*Npix)),dtype=np.int32)
            elif IFD[258][2][0] == 32:
                tifType = 'PE4k'
                pixy = [100.,100.]
                File.seek(8)
                print ('Read PE 4Kx4K tiff file: '+filename)
                image = np.array(np.frombuffer(File.read(4*Npix),dtype=np.float32)/2.**4,dtype=np.int32)
        elif IFD[273][2][0] == 4096:
            tifType = 'Rayonix'
            pixy = [73.242,73.242]
            File.seek(4096)
            print ('Read Rayonix MX300HE tiff file: '+filename)
            image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.uint16),dtype=np.int32)
    elif sizexy == [391,380]:
        pixy = [109.92,109.92]
        File.seek(8)
        image = np.array(np.frombuffer(File.read(2*Npix),dtype=np.int16),dtype=np.int32)
    elif sizexy == [380,391]:
        File.seek(110)
        pixy = [109.92,109.92]
        image = np.array(np.frombuffer(File.read(Npix),dtype=np.uint8),dtype=np.int32)
    elif sizexy ==  [825,830]:
        pixy = [109.92,109.92]
        File.seek(8)
        image = np.array(np.frombuffer(File.read(Npix),dtype=np.uint8),dtype=np.int32)
    elif sizexy ==  [1800,1800]:
        pixy = [109.92,109.92]
        File.seek(110)
        image = np.array(np.frombuffer(File.read(Npix),dtype=np.uint8),dtype=np.int32)
    elif sizexy == [2880,2880]:
        pixy = [150.,150.]
        File.seek(8)
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder(byteOrd)
        image = np.array(np.frombuffer(File.read(Npix*4),dtype=dt),dtype=np.int32)
    elif sizexy == [3070,1102]:
        print ('Read Dectris Eiger 1M tiff file: '+filename)
        pixy = [75.,75.]
        File.seek(8)
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder(byteOrd)
        image = np.array(np.frombuffer(File.read(Npix*4),dtype=np.uint32),dtype=np.int32)
    elif sizexy == [1024,402]:
        pixy = [56.,56.]
        File.seek(8)
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder(byteOrd)
        image = np.array(np.frombuffer(File.read(Npix*2),dtype=np.uint16),dtype=np.int32)
        
#    elif sizexy == [960,960]:
#        tiftype = 'PE-BE'
#        pixy = (200,200)
#        File.seek(8)
#        if not imageOnly:
#            print 'Read Gold tiff file:',filename
#        image = np.array(ar.array('H',File.read(2*Npix)),dtype=np.int32)
           
    if image is None:
        lines = ['not a known detector tiff file',]
        File.close()    
        return lines,0,0,0
        
    if sizexy[1]*sizexy[0] != image.size: # test is resize is allowed
        lines = ['not a known detector tiff file',]
        File.close()    
        return lines,0,0,0
#    if GSASIIpath.GetConfigValue('debug'):
    if DEBUG:
        print ('image read time: %.3f'%(time.time()-time0))
    image = np.reshape(image,(sizexy[1],sizexy[0]))
    center = (not center[0]) and [pixy[0]*sizexy[0]/2000,pixy[1]*sizexy[1]/2000] or center
    wavelength = (not wavelength) and 0.10 or wavelength
    distance = (not distance) and 100.0 or distance
    polarization = (not polarization) and 0.99 or polarization
    samplechangerpos = (not samplechangerpos) and 0.0 or samplechangerpos
    if xpixelsize is not None and ypixelsize is not None:
        pixy_meta = [xpixelsize,ypixelsize]
        # if GSASIIpath.GetConfigValue('debug'):
        if DEBUG:
            print ('pixel size from metadata: '+str(pixy_meta))
        if pixy is None:
            pixy = pixy_meta
    data = {'pixelSize':pixy,'pixelSize_metadata':pixy_meta,'wavelength':wavelength,'distance':distance,'center':center,'size':sizexy,
            'setdist':distance,'PolaVal':[polarization,False],'samplechangerpos':samplechangerpos,'det2theta':0.0}
    File.close()
    return head,data,Npix,image