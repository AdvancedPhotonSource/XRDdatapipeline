import numpy as np


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
    image_f = np.array(image_f * flatfield, dtype=np.int32)
    return image_f


def tth_to_q(tth, wavelength):
    return 4 * np.pi * np.sin(tth / 2 * np.pi / 180) / wavelength


def get_Qbands(Qmap, LUtth, wavelength, numChans):
    Qmin = tth_to_q(LUtth[0], wavelength)
    Qmax = tth_to_q(LUtth[1], wavelength)
    dQ = (Qmax - Qmin) / numChans
    # Qband = np.array(Qmap / dQ, dtype = np.int32) # incorrect, doesn't start at qmin; check tthband
    Qband = np.array((Qmap - Qmin) / dQ, dtype = np.int32)
    bin_edges = np.arange(Qmin, Qmax+dQ, dQ)
    # tth_delta = (tth_max - tth_min) / numChans
    # tth_list = np.arange(tth_min, tth_max + tth_delta / 2.0, tth_delta)
    # tth_val = ((tth_list[1:] + tth_list[:-1]) / 2.0).astype(np.float32)
    return Qband, bin_edges


def Qmap(Tmap, wavelength):
    return 4 * np.pi * np.sin(Tmap / 2 * np.pi / 180) / wavelength


