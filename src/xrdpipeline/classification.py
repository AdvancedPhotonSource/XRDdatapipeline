import time

import numpy as np
import pandas as pd
import skimage as ski
from spottiness import h_maxima_calc, spottiness


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


def modulo_range(array, center, range):
    diff = (array - center) % 360
    # range = range.values
    return np.logical_or(diff < range, diff > (360 - range))


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


def split_grad_with_Q(
    image,
    om,
    arc_mask,
    gradient_dict,
    qmap,
    azmap,
    Qbins,
    threshold_percentile=0.1,
    return_grad=False,
    return_partials=False,
    interpolate=False,
    calc_spottiness=False,
    predef_mask=None,
    predef_mask_extended=None,
):
    print(f"{interpolate=}")
    time0 = time.time()
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
    time1 = time.time()
    print(f"Grad calc time: {time1-time0}")
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
    time2 = time.time()
    print(f"Props table time: {time2-time1}")
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
    time3 = time.time()
    print(f"Edits to table: {time3-time2}")
    # spot_mask = np.in1d(new_mask,spot_clusters)
    Qgrad_arc_mask = np.in1d(labeled_mask, arc_cluster_indices)
    # spot_mask = np.reshape(spot_mask,om.shape)
    Qgrad_arc_mask = np.reshape(Qgrad_arc_mask, om.shape)

    gradient_mask = azim_grad_2 < threshold
    # only look at the areas overlapping potential arcs
    gradient_mask = np.logical_and(gradient_mask, Qgrad_arc_mask)
    # if there aren't any clusters, just skip this part entirely. Maximum will be False if nothing is there.
    time4 = time.time()
    print(f"Gradient mask calc'd: {time4-time3}")
    if np.max(gradient_mask) > 0:
        time_0 = time.time()
        labeled_gradients = ski.measure.label(gradient_mask)
        gradient_props = ["label", "centroid", "coords", "area"]
        gradient_props_table = ski.measure.regionprops_table(
            labeled_gradients, properties=gradient_props
        )
        gradient_props_table = pd.DataFrame(gradient_props_table)
        time_1 = time.time()
        print(f"Label gradient mask: {time_1-time_0}")

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
        time_2 = time.time()
        print(f"Extra table additions: {time_2-time_1}")

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
        time_3 = time.time()
        print(f"Centroid off mask: {time_3-time_2}")

        gradient_props_table.to_csv("gradient_props_table.csv")
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
        time_4 = time.time()
        print(f"Finding azim sections to cut p1: {time_4-time_3}")
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
        time_5 = time.time()
        print(f"Finding azim sections to cut: {time_5-time_4}")
        # def modulo_range(array, center, range):
        #     diff = (array - center) % 360
        #     # range = range.values
        #     return np.logical_or(diff < range, diff > (360 - range))
        # diffs = (azmap[:,:,np.newaxis] * np.ones_like(gradient_props_table["azim_centroid"].values)[np.newaxis,np.newaxis,:] - gradient_props_table["azim_centroid"].values) % 360
        # diffs_low = diffs < gradient_props_table["azim_width"].values
        # diffs_high = diffs > (360 - gradient_props_table["azim_width"].values)
        # diffs = np.logical_or(diffs_low, diffs_high)
        # testing_cuts = np.logical_and(labeled_mask == gradient_props_table["cluster_label"].values, diffs)
        time_6 = time.time()
        # print(f"Alternate azim sections to cut p1: {time_6-time_5}")

        cuts = np.sum(gradient_props_table["new_cuts"].values, axis=0).astype(bool)

        # cut section of gradient mask from arc mask
        new_arc_mask = np.logical_and(Qgrad_arc_mask, ~cuts)
        new_spot_mask = np.logical_and(om, ~new_arc_mask)
        time_7 = time.time()
        print(f"Cutting azim sections: {time_7-time_6}")

    else:
        # print("No spots to cut out. Skipping last step.")
        new_arc_mask = Qgrad_arc_mask
        new_spot_mask = np.logical_and(om, ~new_arc_mask)
    time5 = time.time()
    print(f"Split spot sections: {time5-time4}")
    # spots stats
    # only arcs have stayed in one table; spots have been scattered across multiple
    # so spots stats table needs to be recreated at this point
    spots_props = ["label", "area", "intensity_mean"]
    labeled_spots = ski.measure.label(new_spot_mask)
    spots_table = ski.measure.regionprops_table(
        labeled_spots, qmap, properties=spots_props
    )
    spots_table = pd.DataFrame(spots_table)

    if calc_spottiness:
        reduced_labeled_new_spot_mask = ski.morphology.remove_small_objects(
            labeled_spots, min_size=10
        )
        reduced_new_spot_mask = reduced_labeled_new_spot_mask > 0
        time6 = time.time()
        print(f"New spot table: {time6-time5}")
        # Qbins = get_Qbands(qmap, LUtth, wavelength, numChans)
        percents, num_spots = spottiness(
            reduced_new_spot_mask, reduced_labeled_new_spot_mask, Qbins
        )
        time7 = time.time()
        print(f"Spottiness calculated: {time7-time6}")
        num_maxima, num_spot_maxima = h_maxima_calc(image, reduced_new_spot_mask, Qbins)
        time8 = time.time()
        print(f"h_maxima calculated: {time8-time7}")

    to_return = [new_spot_mask, new_arc_mask, spots_table]

    if return_grad:
        to_return.append(radial_grad_2)
        to_return.append(azim_grad_2)

    if return_partials:
        to_return.append(Qgrad_arc_mask)

    if calc_spottiness:
        to_return.append(percents)
        to_return.append(num_spots)
        to_return.append(num_maxima)
        to_return.append(num_spot_maxima)

    return to_return


def qwidth_area_classification_2(
    om,
    Qmap,
    azmap,
    min_arc_area=100,
    Q_max=0.1,  # 0.08
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
    reduced_om = ski.morphology.remove_small_objects(om, min_arc_area)
    labeled_mask = ski.measure.label(reduced_om)
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
    # small_clusters = props_table.loc[:, "area"].values < min_arc_area
    # small_cluster_indices = props_table.iloc[small_clusters].index
    # props_table.drop(small_cluster_indices, axis=0, inplace=True)

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
        # arcs_bool = np.logical_and(
        #     props_table.loc[:, "area_over_width2"].values > area_Q_shape_min, arcs_bool
        # )
        arcs_bool = np.logical_and(
            props_table.loc[:, "azim_vs_Q"].values > azim_Q_shape_min, arcs_bool
        )
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


def qwidth_area_classification(
    om,
    Qmap,
    azmap,
    min_arc_area=100,
    Q_max=0.1,  # 0.08
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

    # props_table = ski.measure.regionprops_table(
    #     labeled_mask, intensity_image=Qmap, properties=props
    # )
    # props_table = pd.DataFrame(props_table)
    # props_table["diffs_Q"] = props_table["intensity_max"] - props_table["intensity_min"]
    props_table = ski.measure.regionprops_table(
        labeled_mask,
        intensity_image=azmap,
        # properties=["label", "intensity_min", "intensity_max"],
        properties=props,
    )
    props_table = pd.DataFrame(props_table)
    props_table["diffs_azim"] = (
        props_table["intensity_max"] - props_table["intensity_min"]
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
        # arcs_bool = np.logical_and(
        #     props_table.loc[:, "area_over_width2"].values > area_Q_shape_min, arcs_bool
        # )
        arcs_bool = np.logical_and(
            props_table.loc[:, "azim_vs_Q"].values > azim_Q_shape_min, arcs_bool
        )
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


def current_splitting_method(
    image,
    om,
    qmap,
    azmap,
    gradient_dict,
    Qbins,
    threshold_percentile=0.1,
    return_steps=False,
    interpolate=False,
    calc_spottiness=False,
    azim_Q_shape_min=100,
    predef_mask=None,
    predef_mask_extended=None,
):
    # t0 = time.time()
    # base_spot, base_arc = qwidth_area_classification(om, qmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_max = 0.00001)
    # base_spot, base_arc = qwidth_area_classification(om, qmap, azmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_min = 3500)
    time0 = time.time()
    base_spot, base_arc = qwidth_area_classification(
        om,
        qmap,
        azmap,
        min_arc_area=100,
        Q_max=0.08,
        azim_min=3.5,
        azim_Q_shape_min=azim_Q_shape_min,
        compare_shape=True,
    )
    time1 = time.time()
    print(f"Time for qwidth area classification: {time1-time0}")
    print(f"{return_steps=}")
    # t1 = time.time()
    # print("Time to do initial q width / area classification: {0:.2f}s".format(t1-t0))
    # spot_mask, arc_mask = split_grad(image,om,base_arc,gradient_dict,threshold_percentile=threshold_percentile)
    returned_values = split_grad_with_Q(
        image,
        om,
        base_arc,
        gradient_dict,
        qmap,
        azmap,
        Qbins,
        threshold_percentile=threshold_percentile,
        return_partials=return_steps,
        return_grad=return_steps,
        interpolate=interpolate,
        calc_spottiness=calc_spottiness,
        predef_mask=predef_mask,
        predef_mask_extended=predef_mask_extended,
    )
    # can just skip to returning this^
    if return_steps and calc_spottiness:
        (
            spot_mask,
            arc_mask,
            spots_table,
            radial_grad_2,
            azim_grad_2,
            qgrad_arc_mask,
            percents,
            num_spots,
            num_maxima,
            num_spot_maxima,
        ) = returned_values
    elif return_steps:
        spot_mask, arc_mask, spots_table, radial_grad_2, azim_grad_2, qgrad_arc_mask = (
            returned_values
        )
    elif calc_spottiness:
        (
            spot_mask,
            arc_mask,
            spots_table,
            percents,
            num_spots,
            num_maxima,
            num_spot_maxima,
        ) = returned_values
    else:
        spot_mask, arc_mask, spots_table = returned_values
    time2 = time.time()
    print(f"Time for grad splitting: {time2-time1}")
    to_return = [spot_mask, arc_mask, spots_table]
    if return_steps:
        to_return.append(base_arc)
        to_return.append(qgrad_arc_mask)
        to_return.append(azim_grad_2)
        to_return.append(radial_grad_2)
    if calc_spottiness:
        to_return.append(percents)
        to_return.append(num_spots)
        to_return.append(num_maxima)
        to_return.append(num_spot_maxima)
    return to_return
