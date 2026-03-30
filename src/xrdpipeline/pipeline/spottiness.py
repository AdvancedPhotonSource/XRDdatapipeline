"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the spottiness routines for the analysis pipeline.
"""

import numpy as np
import pandas as pd
from skimage.morphology.extrema import h_maxima
from scipy.ndimage import label
import time


def spottiness_df_stats(df,raveled_mask,spot_mask, qbins):
    """
    Calculate area, intensity, and position statistics for the set of
    all spot-classified clusters. Takes less than .1s to run.

    :param df: Pandas dataframe of classification info.
    :param raveled_mask: 1d raveled outlier mask
    :param spot_mask: 2d spot mask
    :param qbins: 2d array of Q bin indices for each pixel
    """
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
    if "on_arc" in df.columns:
        spot_stat["on_arc"] = df.groupby("spot_stat_label")["on_arc"].max()
    return spot_stat


def spottiness_azim_grad(azim_grad_2, qbins):
    """
    Calculates mean, median, standard deviation, and median absolute deviation
    statistics for each Q bin of the second azimuthal derivative. These can be
    used to look at how spotty a particular bin is. Takes 1-2s to run.

    :param azim_grad_2: 2d array of second azimuthal derivative values for the image
    :param qbins: 2d array of Q bin indices for each pixel
    """
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

