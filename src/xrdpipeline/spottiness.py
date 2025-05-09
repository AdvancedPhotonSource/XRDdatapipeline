import numpy as np
from skimage.morphology.extrema import h_maxima
import time

def spottiness(unlabeled_spot_mask, labeled_spot_mask, binsmap):
    time0 = time.time()
    total_pixels = np.array([np.sum(binsmap == i) for i in range(np.max(binsmap))])
    masked = np.ma.array(binsmap, mask = unlabeled_spot_mask)
    unmasked_pixels = np.array([np.ma.sum(masked == i) for i in range(np.max(binsmap))])
    percent_masked = (total_pixels - unmasked_pixels) / total_pixels
    time1 = time.time()
    print(f"Spottiness: Percent masked: {time1-time0}")
    num_unique_spots = np.array([len(np.unique(labeled_spot_mask[binsmap == i])) - 1 for i in range(np.max(binsmap))]) # subtract off 0 value
    time2 = time.time()
    print(f"Spottiness: num unique spots: {time2-time1}")
    return percent_masked, num_unique_spots


def h_maxima_calc(image, spot_mask, binsmap):
    time0 = time.time()
    h = int(0.05*np.percentile(image,99.9))
    image_maxima = h_maxima(image,h)
    time1 = time.time()
    print(f"Actual h_maxima function: {time1-time0}")
    # h_maxima returns a binary array
    spot_h_maxima = np.logical_and(image_maxima, spot_mask)
    # masked = np.ma.array(binsmap, mask = ~h_maxima)
    # binned_maxima = np.array([np.ma.sum(masked == i) for i in range(np.max(binsmap))])
    # masked = np.ma.array(binsmap, mask = ~spot_h_maxima)
    # binned_spot_maxima = np.array([np.ma.sum(masked == i) for i in range(np.max(binsmap))])
    masked = np.array(binsmap)
    masked[~image_maxima] = 0
    binned_maxima = np.array([np.sum(masked == i) for i in range(np.max(binsmap))])
    masked = np.array(binsmap)
    masked[~spot_h_maxima] = 0
    binned_spot_maxima = np.array([np.sum(masked == i) for i in range(np.max(binsmap))])
    time2 = time.time()
    print(f"Binning for h_maxima: {time2-time1}")
    return binned_maxima, binned_spot_maxima

