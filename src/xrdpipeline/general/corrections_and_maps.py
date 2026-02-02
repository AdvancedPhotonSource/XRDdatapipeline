"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the correction and mapping routines for the analysis pipeline.
"""

import numpy as np


# nonzeromask
def nonzeromask(image, mask_negative=True):
    """
    Find all zero and, optionally, negative values in an image and mask them.
    Always called as a basic mask on each image, this gets combined with the
    predefined experimental mask at the start of the process.

    :param image: Image array to check
    :param mask_negative: Also mask negative values
    """
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
        image_p / polmap
    )  # polmap showing high values in center column dropping to low values at right/left edges
    return image_p


# flat-field correction
def flatfield_correct(image, flatfield):
    image_f = np.array(image)
    image_f = np.array(image_f * flatfield)
    return image_f


def tth_to_q(tth, wavelength):
    return 4 * np.pi * np.sin(tth / 2 * np.pi / 180) / wavelength


def q_to_tth(q, wavelength):
    return np.arcsin(q * wavelength / (4 * np.pi)) * (360 / np.pi)


def tth_to_d(tth, wavelength):
    return wavelength / (2 * np.sin(tth / 2 * np.pi / 180))


def get_Qbands(Qmap, LUtth, wavelength, numChans):
    """
    Split a 2d map of all Q values into a number of bands based on
    the input two-theta min/max, the wavelength, and the number of
    bins. Returns a 2d array of bin numbers, with 0 as the out-of-bounds
    bin index.

    :param Qmap: 2d array of Q values to bin
    :param LUtth: Iterable of the min then max values in two-theta
    :param wavelength: Wavelength of the beam used, to translate LUtth to Q
    :param numChans: Number of bins to split the image into
    """
    Qmin = tth_to_q(LUtth[0], wavelength)
    Qmax = tth_to_q(LUtth[1], wavelength)
    dQ = (Qmax - Qmin) / numChans
    # Qband = np.array(Qmap / dQ, dtype = np.int32) # incorrect, doesn't start at qmin; check tthband
    Qband = np.array((Qmap - Qmin) / dQ, dtype = np.int16)
    bin_edges = np.arange(Qmin, Qmax+dQ, dQ)
    # tth_delta = (tth_max - tth_min) / numChans
    # tth_list = np.arange(tth_min, tth_max + tth_delta / 2.0, tth_delta)
    # tth_val = ((tth_list[1:] + tth_list[:-1]) / 2.0).astype(np.float32)
    return Qband, bin_edges

