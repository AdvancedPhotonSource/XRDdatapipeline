"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the classification routines for the analysis pipeline.
"""

import numpy as np
import scipy
import pandas as pd
import time
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

from pipeline.spottiness import spottiness_azim_grad, spottiness_df_stats


def remove_overlaps(labeled_cuts, predef_mask):
    """
    Remove any masked pixels which overlap the predefined experimental mask.
    This was called by the spot-cutting algorithm, but is no longer in use.

    :param labeled_cuts: Labeled mask of pixels
    :param predef_mask: Predefined experimental mask
    """
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


def modulo_range(array, center, range):
    """
    Used for finding pixels with an azimuthal value within some range of
    a central value. The modulus is set to 360.

    :param array: Array of values to check
    :param center: Central value to check against
    :param range: Range to check against
    """
    diff = (array - center) % 360
    # range = range.values
    return np.logical_or(diff < range, diff > (360 - range))


def radial_and_azim_gradient(
    image, r_hat, phi_hat, kernel_x, kernel_y, r=True, azim=True
):
    """
    Calculate the radial and azimuthal derivatives of an array of pixels,
    averaged across each neighboring pixel as defined by the kernel.
    The x and y derivatives are calculated first, using the x and y kernels
    for their list of neighbors. This is then converted to the radial and
    azimuthal directions using the r_hat and phi_hat matrices.

    :param image: 2d array of pixel intensities
    :param r_hat: Array of the 2theta-aligned unit vector for each pixel, given in x-y coordinates
    :param phi_hat: Array of the azimuthal unit vector for each pixel, given in x-y coordinates
    :param kernel_x: Convolution kernel for finding the x component of the average gradient calculated from surrounding neighbors
    :param kernel_y: Convolution kernel for finding the y component of the average gradient calculated from surrounding neighbors
    :param r: Whether to calculate and return the 2theta-aligned component of the gradient
    :param azim: Whether to calculate and return the azimuthally-aligned component of the gradient
    """
    # footprint = footprint.astype(np.uint)

    from scipy.ndimage import correlate

    grad_x = correlate(image, kernel_x)
    grad_y = correlate(image, kernel_y)
    # print("Convolutions done")

    grad = np.stack([grad_y, grad_x], axis=0)
    to_return = []
    if r:
        to_return.append(np.einsum("ijk,ijk -> jk", grad, r_hat))
    if azim:
        to_return.append(np.einsum("ijk,ijk -> jk", grad, phi_hat))
    return to_return


def qwidth_area_classification_groupby(
    om,
    image,
    Qmap,
    azmap,
    min_arc_area=100,
    Q_max=0.1, # 0.08
    azim_min=3.5,
    azim_Q_shape_min=100,
):
    """
    Spot/texture classification step which looks at the width in Q and the total
    area of the cluster.

    :param om: Outlier mask
    :param image: Image
    :param Qmap: 2d array of Q values for each pixel
    :param azmap: 2d array of azimuthal values for each pixel
    :param min_arc_area: Minimum pixel area to be considered an arc
    :param Q_max: Maximum Q width to be considered an arc
    :param azim_min: Minimum azimuthal range to be considered an arc
    :param azim_Q_shape_min: Minimum ratio of azimuthal width to Q width to be considered an arc
    """
    flipped_azmap = np.fliplr(azmap)
    labeled_mask, num_features = scipy.ndimage.label(om)
    raveled_labels = labeled_mask.ravel()
    raveled_mask = om.ravel()
    raveled_image = image.ravel()
    raveled_Qmap = Qmap.ravel()
    raveled_azmap = azmap.ravel()
    raveled_flipped_azmap = flipped_azmap.ravel()
    df = pd.DataFrame({
        'label': raveled_labels[raveled_mask],
        'intensity': raveled_image[raveled_mask],
        'Qvalue': raveled_Qmap[raveled_mask],
        'azimvalue': raveled_azmap[raveled_mask],
        'flipped_azimvalue': raveled_flipped_azmap[raveled_mask],
    })
    areas = df['label'].value_counts()
    valid_labels = areas[areas > min_arc_area].index
    max_azim_a = df[df['label'].isin(valid_labels)].groupby('label')['azimvalue'].max()
    min_azim_a = df[df['label'].isin(valid_labels)].groupby('label')['azimvalue'].min()
    diff_azim_a = max_azim_a - min_azim_a
    max_azim_b = df[df['label'].isin(valid_labels)].groupby('label')['flipped_azimvalue'].max()
    min_azim_b = df[df['label'].isin(valid_labels)].groupby('label')['flipped_azimvalue'].min()
    diff_azim_b = max_azim_b - min_azim_b
    diff_azim = pd.concat([diff_azim_a, diff_azim_b]).groupby(level=0).min()

    max90Q = df[df['label'].isin(valid_labels)].groupby('label')['Qvalue'].agg(lambda x: np.percentile(x, 90))
    min10Q = df[df['label'].isin(valid_labels)].groupby('label')['Qvalue'].agg(lambda x: np.percentile(x, 10))
    diff_Q = max90Q - min10Q

    azim_vs_Q = diff_azim / diff_Q

    maxQ_bool = diff_Q < Q_max
    minazim_bool = diff_azim > azim_min
    azim_Q_bool = azim_vs_Q > azim_Q_shape_min
    arcs_bool = maxQ_bool & minazim_bool & azim_Q_bool

    df['classifier'] = np.zeros(len(raveled_labels[raveled_mask]))
    df.loc[df['label'].isin(arcs_bool[arcs_bool].index), 'classifier'] = 2
    df.loc[df['label'].isin(arcs_bool[~arcs_bool].index), 'classifier'] = 1
    
    return df, valid_labels, labeled_mask, raveled_mask


def split_grad_with_Q_groupby(
    image,
    raveled_mask,
    df,
    valid_labels,
    gradient_dict,
    predef,
    labeled_mask,
    threshold_percentile = 0.1,
    report_times = True,
):
    """
    Spot/texture classification step which uses second radial and azimuthal
    derivatives. The latter is used to find spot-like features, and the former
    is used to find the center of an arc.
    The threshold calculation for finding outliers along the azimuthal direction
    is calculated using a percentile of the second radial derivative due to that
    direction being filled with more normal shifts from rings. The azimuthal direction
    is only taken up by spots and texture.

    :param image: 2d image array
    :param raveled_mask: 1d raveled outlier mask
    :param df: Pandas dataframe of classification info from previous steps
    :param valid_labels: Subset of labels for clusters which are large enough
    to be considered for classification
    :param gradient_dict: Dictionary of gradient info
    :param predef: Predefined experimental mask
    :param labeled_mask: Labeled outlier mask
    :param threshold_percentile: Percentile of second radial derivative used to find
    the center of a spot
    :param report_times: Return times taken to perform each step
    """
    if report_times: t0 = time.time()
    if report_times:
        t1 = time.time()
        print(f"Import time: {t1-t0}")

    if report_times: t0 = time.time()
    kernel = Gaussian2DKernel(x_stddev=1)
    image[predef] = np.nan
    interpolated_image = interpolate_replace_nans(image, kernel)
    if report_times:
        t1 = time.time()
        print(f"Interpolation time: {t1-t0}")

    if report_times: t0 = time.time()
    radial_grad, azim_grad = radial_and_azim_gradient(
        interpolated_image,
        gradient_dict["r_hat"],
        gradient_dict["phi_hat"],
        gradient_dict["kernel_x"],
        gradient_dict["kernel_y"],
    )
    azim_grad_2, = radial_and_azim_gradient(
        azim_grad,
        gradient_dict["r_hat"],
        gradient_dict["phi_hat"],
        gradient_dict["kernel_x"],
        gradient_dict["kernel_y"],
        r=False,
    )
    radial_grad_2, = radial_and_azim_gradient(
        radial_grad,
        gradient_dict["r_hat"],
        gradient_dict["phi_hat"],
        gradient_dict["kernel_x"],
        gradient_dict["kernel_y"],
        azim=False,
    )
    if report_times:
        t1 = time.time()
        print(f"Gradient calc time: {t1-t0}")

    if report_times: t0 = time.time()
    non_nan_radial_grad_2 = radial_grad_2.ravel()[~np.isnan(radial_grad_2.ravel())]
    threshold = np.percentile(non_nan_radial_grad_2, threshold_percentile)
    on_arc_threshold = np.percentile(non_nan_radial_grad_2, 10)
    if report_times:
        t1 = time.time()
        print(f"Threshold calc time: {t1-t0}")

    if report_times: t0 = time.time()
    # now want full (not 90th percentile) max, min, as well as median
    maxQ = df[df['label'].isin(valid_labels)].groupby('label')['Qvalue'].max()
    minQ = df[df['label'].isin(valid_labels)].groupby('label')['Qvalue'].min()
    medianQ = df[df['label'].isin(valid_labels)].groupby('label')['Qvalue'].median()
    diffQ = maxQ - minQ
    if report_times:
        t1 = time.time()
        print(f"Min/max/median/diff times: {t1-t0}")

    # find the values of the second radial grad for pixels within 0.02 of the Q median
    # find the 20th percentile of those values
    # find clusters where that percentile is less than the cutoff
    if report_times: t0 = time.time()
    raveled_second_radial = radial_grad_2.ravel()
    df["second_radial"] = raveled_second_radial[raveled_mask]
    if report_times:
        t1 = time.time()
        print(f"Raveling and adding second radial: {t1-t0}")
    if report_times: t0 = time.time()
    df.loc[df['label'].isin(valid_labels),"medianQ"] = medianQ.loc[df.loc[df['label'].isin(valid_labels),'label']].values
    if report_times:
        t1 = time.time()
        print(f"Assigning the median Q values to the table: {t1-t0}")

    if report_times: t0 = time.time()
    high_values = df[df["Qvalue"] > df["medianQ"] - 0.02].index
    low_values = df[df["Qvalue"] < df["medianQ"] + 0.02].index
    central_values = high_values.intersection(low_values)
    df["central_values"] = False # else can't groupby
    df.loc[central_values,"central_values"] = True
    if report_times:
        t1 = time.time()
        print(f"Finding central values: {t1-t0}")

    # on arc by radial grad consideration
    radial_grad_percentile = df[df["central_values"]].groupby("label")["second_radial"].agg(lambda x: np.percentile(x, 20))
    df["on_arc"] = -1
    on_arc = radial_grad_percentile < on_arc_threshold
    # need to extrapolate that info out to all valid_labels, not just central_values
    df.loc[df["label"].isin(valid_labels),"on_arc"] = (on_arc.loc[df.loc[df["label"].isin(valid_labels),"label"]].values) * 1
    
    # azimuthal gradient sections
    azim_gradient_mask = azim_grad_2 < threshold
    # &and this with those labeled as on arc
    azim_gradient_mask = azim_gradient_mask.ravel()
    # df has the raveled_mask applied
    azim_gradient_mask_shortened = azim_gradient_mask[raveled_mask]
    azim_gradient_mask_shortened &= (df["on_arc"].values == 1)
    azim_gradient_mask[raveled_mask] = azim_gradient_mask_shortened

    # if there aren't any clusters, just skip this part entirely. Maximum will be False if nothing is there.
    if np.max(azim_gradient_mask) > 0:
        df = remove_azim_spots_numpy(df, image.shape, raveled_mask, azim_gradient_mask, diffQ, labeled_mask)
    else:
        df["new_arc"] = (df["on_arc"] == 1) & (df["classifier"] == 2)
        df["new_spot"] = (df["on_arc"] == 0) | (df["classifier"] == 1)

    raveled_new_spot = np.zeros_like(raveled_mask)
    raveled_new_spot[raveled_mask] = df["new_spot"].values
    raveled_new_arc = np.zeros_like(raveled_mask)
    raveled_new_arc[raveled_mask] = df["new_arc"].values
    spot_mask = raveled_new_spot.reshape(image.shape)
    arc_mask = raveled_new_arc.reshape(image.shape)

    return spot_mask, arc_mask, df, azim_grad_2

def remove_azim_spots_numpy(df, image_shape, raveled_mask, azim_gradient_mask, diffQ, labeled_mask):
    """
    Function for searching through 2nd azimuthal derivative threshold masks and turning them into
    spots to cut from tagged arc clusters.
    Pure NumPy implementation.

    :param df: Pandas dataframe of classification info from previous steps
    :param image_shape: Tuple of the 2d image shape
    :param raveled_mask: 1d raveled outlier mask
    :param azim_gradient_mask: 1d raveled mask of spot-tagged sections from second azimuthal derivative-based
    identification
    :param diffQ: Widths in Q for each cluster
    :param labeled_mask: 2d outlier mask with a unique label for each cluster
    """
    from scipy import ndimage

    # Setup
    azim_gradient_mask_2d = azim_gradient_mask.reshape(image_shape)
    labeled_gradient_mask, num_gradient_labels = ndimage.label(azim_gradient_mask_2d)
    labeled_gradient_flat = labeled_gradient_mask.ravel()

    newlabels = labeled_gradient_flat[raveled_mask]
    df["newlabel"] = newlabels

    # Extract arrays
    n_pixels = len(df)
    azimvalues = df["azimvalue"].values
    flippedvalues = df["flipped_azimvalue"].values
    labels = df["label"].values
    on_arc = df["on_arc"].values
    classifier = df["classifier"].values

    # Compute Qwidths
    labeled_mask_flat = labeled_mask.ravel()
    max_label = labeled_mask_flat.max()
    diffQ_array = np.zeros(max_label + 1)
    diffQ_array[diffQ.index] = diffQ.values
    Qwidths = diffQ_array[labeled_mask_flat[raveled_mask]]
    df["new_Qwidths"] = Qwidths

    # Initialize output
    close_to_median_azim = np.zeros(n_pixels, dtype=bool)
    close_to_median_azim_flipped = np.zeros(n_pixels, dtype=bool)

    # Process each unique newlabel
    if num_gradient_labels > 0:
        unique_newlabels = np.unique(newlabels[newlabels != 0])

        for nl in unique_newlabels:
            mask_nl = newlabels == nl

            # Compute medians for this newlabel
            median_azim = np.median(azimvalues[mask_nl])
            median_flipped = np.median(flippedvalues[mask_nl])
            parent_label = labels[mask_nl][0]

            # Find all pixels with this parent label
            mask_parent = labels == parent_label

            # Check distances
            azim_diff = np.abs(azimvalues[mask_parent] - median_azim)
            flipped_diff = np.abs(flippedvalues[mask_parent] - median_flipped)
            width_thresh = 5 * Qwidths[mask_parent]

            # Update boolean arrays
            close_azim = azim_diff < width_thresh
            close_flipped = flipped_diff < width_thresh

            # Use |= to accumulate results (in case multiple newlabels per parent)
            close_to_median_azim[mask_parent] |= close_azim
            close_to_median_azim_flipped[mask_parent] |= close_flipped

    df["close_to_median_azim"] = close_to_median_azim
    df["close_to_median_azim_flipped"] = close_to_median_azim_flipped

    # Final classification
    new_arc = (on_arc == 1) & (classifier == 2)
    swap = (
        (close_to_median_azim & (azimvalues > 10) & (azimvalues < 350)) |
        close_to_median_azim_flipped
    )
    new_arc[swap] = False

    new_spot = (on_arc == 0) | (classifier == 1)
    new_spot[swap] = True

    df["new_arc"] = new_arc
    df["new_spot"] = new_spot

    return df


def current_splitting_method(
    image,
    om,
    qmap,
    azmap,
    gradient_dict,
    Qbins,
    threshold_percentile=0.1,
    calc_spot_stats=True,
    calc_grad_spottiness=False,
    azim_Q_shape_min=100,
    predef_mask=None,
    min_arc_area=100,
    timing = None,
    timing_names = None,
):
    """
    Current spot/texture classification method which splits the outlier
    mask into two separate masks.

    :param image: 2d array of pixels
    :param om: Outlier mask
    :param qmap: 2d array of Q values for each pixel
    :param azmap: 2d array of azimuthal values for each pixel
    :param gradient_dict: Dictionary of gradient information
    :param Qbins: 2d array of Q bin values for each pixel
    :param threshold_percentile: Percentile of second radial derivative used to find
    the center of a spot
    :param calc_spot_stats: Calculate and return area, number, and other statistics
    on the spot-classified clusters
    :param calc_grad_spottiness: Calculate and return 2nd derivative information on each Q bin
    :param azim_Q_shape_min: Minimum ratio of azimuthal width to Q width to be considered an arc
    :param predef_mask: Predefined experimental mask
    :param min_arc_area: Minimum pixel area to be considered an arc
    :param timing: Return timing values
    :param timing_names: List of timing checkpoint names to append to
    """
    if timing is not None:
        time0 = time.time()
    df, valid_labels, labeled_mask, raveled_mask = qwidth_area_classification_groupby(
        om,
        image,
        qmap,
        azmap,
        min_arc_area=min_arc_area,
        Q_max=0.1,
        azim_min=3.5,
        azim_Q_shape_min=azim_Q_shape_min,
    )
    if timing is not None:
        time1 = time.time()
        # print(f"Time for qwidth area classification: {time1-time0}")
        timing.append(time1 - time0)
        timing_name = "Shape classification"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        time0 = time.time()
    spot_mask, arc_mask, df, azim_grad_2 = split_grad_with_Q_groupby(
        image,
        raveled_mask,
        df,
        valid_labels,
        gradient_dict,
        predef=predef_mask,
        labeled_mask=labeled_mask,
        threshold_percentile=threshold_percentile,
        report_times=False,
    )
    if timing is not None:
        time1 = time.time()
        # print(f"Time for grad splitting: {time2-time1}")
        timing.append(time1 - time0)
        timing_name = "Gradient classification"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        time0 = time.time()
    # expecting a table of spot stats for the last return value
    # to_return = [spot_mask, arc_mask, df["classifier"]]
    to_return = [spot_mask, arc_mask]

    if calc_spot_stats:
        spot_table_df = spottiness_df_stats(df, raveled_mask, spot_mask, Qbins)
        to_return.append(spot_table_df)
        if timing is not None:
            time1 = time.time()
            # print(f"Time for grad splitting: {time2-time1}")
            timing.append(time1 - time0)
            timing_name = "Spottiness calculation: stats DF"
            if timing_name not in timing_names:
                timing_names.append(timing_name)
    if calc_grad_spottiness:
        spot_table_grad = spottiness_azim_grad(azim_grad_2, Qbins)
        to_return.append(spot_table_grad)
        if timing is not None:
            time1 = time.time()
            # print(f"Time for grad splitting: {time2-time1}")
            timing.append(time1 - time0)
            timing_name = "Spottiness calculation: 2nd azim grad info"
            if timing_name not in timing_names:
                timing_names.append(timing_name)

    return to_return

