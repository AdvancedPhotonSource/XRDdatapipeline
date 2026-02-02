"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the cache creation routine for the analysis pipeline.
"""


from PIL import Image
import torch
import time
import numpy as np
import argparse
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
mid_dir = os.path.split(script_dir)[0]
if mid_dir not in sys.path:
    sys.path.append(mid_dir)

from general.GSASII_imports import *
from general.corrections_and_maps import tth_to_q, get_Qbands, add_output_subdirectory


def prepare_integration_maps(tth_map, pol_map, dist_map, tth_min, tth_max, numChans, logging = False):
    """
    Prepare maps for torch integration

    :param tth_map: 2d array of 2theta values
    :param pol_map: 2d array of polarization values
    :param dist_map: 2d distance map
    :param tth_min: Minimum 2theta integration bound
    :param tth_max: Maximum 2theta integration bound
    :param numChans: Number of integration bins
    :param logging: return timing logs
    """
    if logging:
        t0 = time.time()
    tth = tth_map.ravel()
    raveled_pol = pol_map.ravel()
    raveled_dist = dist_map.ravel() ** 1.5 # the dist map is squared distance, here it needs to be cubic
    if logging:
        t1 = time.time()
        print(f"raveling: {(t1-t0):.2f}")
        t0 = time.time()

    tth_delta = (tth_max - tth_min) / numChans
    tth_list = np.arange(tth_min, tth_max + tth_delta / 2.0, tth_delta)
    tth_val = ((tth_list[1:] + tth_list[:-1]) / 2.0).astype(np.float32)
    if logging:
        t1 = time.time()
        print(f"tth delta, list, val: {(t1-t0):.2f}")
        t0 = time.time()

    tth_idx = np.array((tth - tth_min) / tth_delta, dtype=np.int32)
    tth_idx = np.where(tth_idx < 0, 0, tth_idx)
    tth_idx = np.where(tth_idx > numChans, 0, tth_idx)
    if logging:
        t1 = time.time()
        print(f"tth idx map: {(t1-t0):.2f}")
        t0 = time.time()

    tth_idx = torch.from_numpy(tth_idx)
    tth_val = torch.from_numpy(tth_val)
    if logging:
        t1 = time.time()
        print(f"numpy -> torch: {(t1-t0):.2f}")
        t0 = time.time()
    return tth_idx, tth_val, raveled_pol, raveled_dist, len(tth_val)


# create and save TA[x] maps
# from savemaps.py by Wenqian Xu
def getmaps(
    cache, imctrls, imctrlname, pathmaps
):  # fast integration using the same imctrl and mask
    """
    Get output 2theta and azimuth maps from the GSASIIscriptable function,
    calculate the corresponding Q map, and save all to disk and cache

    :param cache: Dictionary to save to
    :param imctrls: Dictionary of image controls
    :param imctrlname: Name of image control file. Used for the output file names.
    :param pathmaps: Directory path to save to
    """
    TA = Make2ThetaAzimuthMap(imctrls, (0, imctrls["size"][0]), (0, imctrls["size"][1]))
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
    Qmap = tth_to_q(TA[0], imctrls["wavelength"])
    im = Image.fromarray(Qmap)
    im.save(os.path.splitext(path1)[0] + "_qmap.tif")
    cache["pixelQmap"] = Qmap
    return


def get_azimbands(azmap, numChansAzim):
    """
    Create and return an array of azimuthal band indices given
    a 2d array of azimuthal values and a number of bins.

    :param azmap: 2d array of azimuthal values in degrees
    :param numChansAzim: Number of bins
    """
    dazim = (360) / numChansAzim
    azimband = np.array(azmap / dazim, dtype=np.int16)
    return azimband


def r_and_phi_hat(image_shape, center):
    """
    Calculate and return the r-hat and phi-hat vectors for each
    pixel in an image with given shape and center.
    This is used when calculating the radial and azimuthal derivatives.

    :param image_shape: Shape of the image
    :param center: Center of the image in pixel units
    """
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
    """
    Calculates the r-hat, phi-hat, and x and y kernels used for gradient
    calculation. Returns a dict of these arrays.
    The kernels allow derivative calculations to be performed as convolutions,
    and the r-hat and phi-hat arrays convert the representation of the gradient
    from x-y to r-phi.

    :param image_shape: Shape of the image
    :param center: Center of the image in pixel coordinates
    :param footprint: 2d boolean array defining neighboring pixels for average
    gradient calculation
    """
    # calculate distances and x-y-basis angles once for each pixel in footprint
    t0 = time.time()
    if not all([i % 2 == 1 for i in footprint.shape]):
        raise ValueError("Footprint shape must be odd in each direction.")
    central_footprint_point = np.array([i // 2 for i in footprint.shape])
    footprint[central_footprint_point[0], central_footprint_point[1]] = 0
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
    r_hat, phi_hat = r_and_phi_hat(image_shape, center)
    t1 = time.time()
    print(
        "Time spent on gradient cache calculations: {0:.2f}s".format(
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

def create_cache(
        cache,
        filename,
        imctrlname,
        output_directory,
        tth_integration_range=None,
        azim_integration_range=None,
        n_integration_bins=None,
        polarization=None,
        imgmaskname = None,
        bad_pixels = None,
        flatfield = None,
        esdMul = 3.0,
        cache_location = None,
        verbose = False,
):
    output_directory = add_output_subdirectory(output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    newdirs = ["maps"]
    for newdir in newdirs:
        path = os.path.join(output_directory, newdir)
        if not os.path.exists(path):
            os.mkdir(path)

    if verbose:
        print("Creating cache")
        t0 = time.time()
    image_dict = read_image(filename)
    if verbose:
        t1 = time.time()
        print(f"read_image(): {(t1-t0):.2f}")
        t0 = time.time()
    if os.path.splitext(imctrlname)[1] == ".imctrl":
        with open(imctrlname, "r") as imctrlfile:
            lines = imctrlfile.readlines()
            LoadControls(lines, image_dict["Image Controls"])
    else:
        with open(imctrlname, "r") as imctrlfile:
            lines = imctrlfile.readlines()
            LoadControlsPONI(lines, image_dict["Image Controls"])
    if tth_integration_range is not None:
        image_dict["Image Controls"]["IOtth"] = tth_integration_range
    if azim_integration_range is not None:
        image_dict["Image Controls"]["LRazimuth"] = azim_integration_range
    if n_integration_bins is not None:
        image_dict["Image Controls"]["outChannels"] = n_integration_bins
    if polarization is not None:
        # check matches format [float, bool]
        image_dict["Image Controls"]["PolaVal"] = polarization
    if verbose:
        t1 = time.time()
        print(f"LoadControls(): {(t1-t0):.2f}")
        t0 = time.time()
    # cache["image"] = load_image(filename)
    if verbose:
        t1 = time.time()
        print(f"load_image(): {(t1-t0):.2f}")
        t0 = time.time()

    predef_mask = {}
    save_predef = False
    if (imgmaskname is not None) and (imgmaskname != ""):
        # img.loadMasks(imgmaskname)
        suffix = imgmaskname.split(".")[1]
        if suffix == "immask":
            readMasks(imgmaskname, image_dict["Masks"], False)
        elif suffix == "tif":
            predef_mask = read_image(imgmaskname)
        save_predef = True
    else:
        predef_mask["image"] = np.zeros_like(image_dict["image"], dtype=bool)
    if (bad_pixels is not None) and (bad_pixels != ""):
        suffix = bad_pixels.split(".")[1]
        if suffix == "tif":
            bad_pixel_mask = read_image(bad_pixels)
            predef_mask |= bad_pixel_mask
        else:
            print("Unsupported bad pixel mask image type. Skipping file read. Any zero-intensity pixels will automatically be masked.")
    cache["predef_mask"] = predef_mask

    flatfield_image = None
    if (flatfield is not None) and (flatfield != ""):
        flatfield_image = load_image(flatfield)
    cache["flatfield"] = flatfield_image
    if verbose:
        t1 = time.time()
        print(f"predef, bad pixel, flatfield: {(t1-t0):.2f}")
        t0 = time.time()

    if save_predef:
        imsave = Image.fromarray(predef_mask["image"])
        imsave.save(
            os.path.join(
                output_directory,
                "maps",
                os.path.splitext(os.path.split(imgmaskname)[1])[0] + ".tif"
            )
        )
    if (flatfield is not None) and (flatfield != ""):
        imsave = Image.fromarray(flatfield_image)
        imsave.save(
            os.path.join(
                output_directory,
                "maps",
                os.path.splitext(os.path.split(flatfield)[1])[0] + ".tif"
            )
        )
    if verbose:
        t1 = time.time()
        print(f"predef, flatfield save: {(t1-t0):.2f}")
        t0 = time.time()
    if verbose:
        t1 = time.time()
        print(f"Image controls: {(t1-t0):.2f}")
        t0 = time.time()
    _, tifdata, _, _ = GetTifData(filename)
    image_dict["Image Controls"]["pixelSize"] = tifdata["pixelSize"]
    if verbose:
        t1 = time.time()
        print(f"GetTifData(): {(t1-t0):.2f}")
        t0 = time.time()

    getmaps(cache, image_dict["Image Controls"], imctrlname, os.path.join(output_directory, "maps"))
    if verbose:
        t1 = time.time()
        print(f"getmaps(): {(t1-t0):.2f}")
        t0 = time.time()
    cache["AzimMask"] = np.logical_or(
        cache["pixelAzmap"] < image_dict["Image Controls"]["LRazimuth"][0],
        cache["pixelAzmap"] > image_dict["Image Controls"]["LRazimuth"][1]
        )
    if verbose:
        t1 = time.time()
        print(f"AzimMask: {(t1-t0):.2f}")
        t0 = time.time()
    # 2th fairly linear along center; calc 2th - pixelsize conversion
    center = image_dict["Image Controls"]["center"]
    center[0] = center[0] * 1000.0 / image_dict["Image Controls"]["pixelSize"][0]
    center[1] = center[1] * 1000.0 / image_dict["Image Controls"]["pixelSize"][1]
    image_dict["center"] = center
    cache["esdMul"] = esdMul
    image_dict["Masks"]["SpotMask"]["esdMul"] = esdMul
    if verbose:
        t1 = time.time()
        print(f"pix size, center, esdMul: {(t1-t0):.2f}")
        t0 = time.time()
    numChansAzim = 360
    cache["azimband"] = get_azimbands(cache["pixelAzmap"], numChansAzim)
    if verbose:
        t1 = time.time()
        print(f"get_azimbands(): {(t1-t0):.2f}")
        t0 = time.time()

    # numChans
    LUtth = np.array(image_dict["Image Controls"]["IOtth"])
    wave = image_dict["Image Controls"]["wavelength"]
    dsp0 = wave / (2.0 * sind(LUtth[0] / 2.0))
    dsp1 = wave / (2.0 * sind(LUtth[1] / 2.0))
    x0 = GetDetectorXY2(dsp0, 0.0, image_dict["Image Controls"])[0]
    x1 = GetDetectorXY2(dsp1, 0.0, image_dict["Image Controls"])[0]
    if not np.any(x0) or not np.any(x1):
        raise Exception
    numChans = int(1000 * (x1 - x0) / image_dict["Image Controls"]["pixelSize"][0]) // 2
    cache["numChans"] = numChans
    if verbose:
        t1 = time.time()
        print(f"numChans: {(t1-t0):.2f}")
        t0 = time.time()
    cache["Qbins"], cache["QbinEdges"] = get_Qbands(cache["pixelQmap"], LUtth, wave, numChans)
    if verbose:
        t1 = time.time()
        print(f"get_Qbands(): {(t1-t0):.2f}")
        t0 = time.time()

    # pytorch integration
    (
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    ) = prepare_integration_maps(
        cache["pixelTAmap"],
        cache["polscalemap"],
        cache["pixelsampledistmap"],
        image_dict["Image Controls"]["IOtth"][0],
        image_dict["Image Controls"]["IOtth"][1],
        image_dict["Image Controls"]["outChannels"],
    )

    if verbose:
        t1 = time.time()
        print(f"prepare_qmaps(): {(t1-t0):.2f}")
        t0 = time.time()

    # gradient info
    cache["gradient"] = gradient_cache(
        predef_mask["image"].shape, center, np.ones((3, 3), dtype=np.uint)
    )
    if verbose:
        t1 = time.time()
        print(f"gradient_cache(): {(t1-t0):.2f}")
        t0 = time.time()

    # store this in cache to include corrections made
    cache["image_dict"] = image_dict
    if verbose:
        t1 = time.time()
        print(f"image_dict: {(t1-t0):.2f}")
        t0 = time.time()

    if cache_location is None:
        cache_location = os.path.join(
            output_directory,
            "maps",
            os.path.splitext(os.path.split(imctrlname)[1])[0]
        )
        cache_location_append = f"_iotth_{image_dict['Image Controls']['IOtth'][0]}_{image_dict['Image Controls']['IOtth'][1]}"
        cache_location_append += f"_LRazimuth_{image_dict['Image Controls']['LRazimuth'][0]}_{image_dict['Image Controls']['LRazimuth'][1]}"
        cache_location_append += f"_outChannels_{image_dict['Image Controls']['outChannels']}"
        cache_location_append += f"_PolaVal_{image_dict['Image Controls']['PolaVal'][0]}"
        cache_location_append += f"_esdMul_{cache['esdMul']}"
        cache_location_append = cache_location_append.replace(".","p")
        cache_location += cache_location_append + ".npy"

    np.save(cache_location, cache)
    if verbose:
        print(f"Size of cache: {sys.getsizeof(cache)}")
        print(cache.keys())
        for k, v in cache.items():
            print(f"{k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["gradient"].items():
            print(f"gradient {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"].items():
            print(f"image_dict {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"]["Comments"].items():
            print(f"image_dict Comments {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"]["Image Controls"].items():
            print(f"image_dict Image Controls {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"]["Masks"].items():
            print(f"image_dict Masks {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"]["Masks"]["SpotMask"].items():
            print(f"image_dict Masks SpotMask {k}: {sys.getsizeof(v)}, {type(v)}")
        for k, v in cache["image_dict"]["Stress/Strain"].items():
            print(f"image_dict Stress/Strain {k}: {sys.getsizeof(v)}, {type(v)}")

    return cache_location, cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--filename", help="Input file", required=True)
    parser.add_argument("-o", "--output_directory", required=True, help="Location to place the output files from this pipeline")
    parser.add_argument("-c", "--imctrl", required=True, help="Image control file")
    parser.add_argument("-f", "--flatfield", help="Flatfield file")
    parser.add_argument("-m", "--imgmask", help="Experimental mask")
    parser.add_argument("-b", "--bad_pixels", help="Detector known bad pixel mask")
    parser.add_argument("-t", "--tth_integration_range", nargs=2, type=float, help= "2theta integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("-z", "--azim_integration_range", nargs=2, type=float, help="Azimuthal integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("--n_integration_bins", type=float, help="Number of bins to use for integration, if overriding or not included in the config file.")
    parser.add_argument("-p", "--polarization", type=float, help="Polarization of the image")
    parser.add_argument("--outlier_mad_mult", type=float, default=3.0, help="Multiplier of median absolute deviation to use when considering a value an outlier. Default is 3.")
    parser.add_argument("-l", "--cache_location", help="Output location to place the cache. If left as None, this will use a default location inside the output directory with a name based on all arguments. Recommended to leave as None.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print extra logging statements.")

    args = parser.parse_args()

    create_cache(
        cache={},
        filename=args.filename,
        imctrlname=args.imctrl,
        output_directory=args.output_directory,
        tth_integration_range=args.tth_integration_range,
        azim_integration_range=args.azim_integration_range,
        n_integration_bins=args.n_integration_bins,
        polarization=args.polarization,
        imgmaskname = args.imgmask,
        bad_pixels = args.bad_pixels,
        flatfield = args.flatfield,
        esdMul = args.outlier_mad_mult,
        cache_location = args.cache_location,
        verbose = args.verbose,
    )
