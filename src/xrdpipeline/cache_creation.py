"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the cache creation routine for the analysis pipeline.
"""


from PIL import Image
from GSASII_imports import *
import torch
import time
from corrections_and_maps import tth_to_q


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
    azimband = np.array(azmap / dazim, dtype=np.int32)
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
