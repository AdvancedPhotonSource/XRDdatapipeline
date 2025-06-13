import numpy as np
import pandas as pd
from skimage.morphology.extrema import h_maxima
from scipy.ndimage import label
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


def spottiness_df_stats(df,raveled_mask,spot_mask, qbins):
    labeled_spot_mask, numlabels = label(spot_mask)
    labeled_spot_mask = labeled_spot_mask.ravel()
    df["spot_stat_label"] = labeled_spot_mask[raveled_mask]
    df["Qbin"] = qbins.ravel()[raveled_mask]
    spot_stat = pd.DataFrame()
    spot_stat["area"] = df["spot_stat_label"].value_counts().sort_index()
    spot_stat["medianQ"] = df.groupby("spot_stat_label")["Qvalue"].median()
    spot_stat["Qbin"] = df.groupby("spot_stat_label")["Qbin"].median().astype(int)
    spot_stat["intensity_sum"] = df.groupby("spot_stat_label")["intensity"].sum()
    spot_stat["intensity_max"] = df.groupby("spot_stat_label")["intensity"].max()
    spot_stat["intensity_mean"] = df.groupby("spot_stat_label")["intensity"].mean()
    # following is only True for those near the center of an arc, not spot
    # actually, looks like on_arc is calculated for all clusters
    # this does require the specific newly-labeled spot cluster to have a section near the center
    spot_stat["on_arc"] = df.groupby("spot_stat_label")["on_arc"].max()
    return spot_stat


def spottiness_azim_grad(azim_grad_2, qbins):
    grad_info = pd.DataFrame({
        "azim_grad_2": azim_grad_2.ravel(),
        "Qbin": qbins.ravel()
    })
    spot_stat = pd.DataFrame()
    spot_stat["mean"] = grad_info.groupby("Qbin")["azim_grad_2"].mean()
    spot_stat["std"] = grad_info.groupby("Qbin")["azim_grad_2"].std()
    spot_stat["median"] = grad_info.groupby("Qbin")["azim_grad_2"].median()
    grad_info["median"] = spot_stat.loc[grad_info["Qbin"],"median"].values
    grad_info["abs_dev"] = np.abs(grad_info["median"] - grad_info["azim_grad_2"])
    spot_stat["mad"] = grad_info.groupby("Qbin")["abs_dev"].median()
    return spot_stat


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

