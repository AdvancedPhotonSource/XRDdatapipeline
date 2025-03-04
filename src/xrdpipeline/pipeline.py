import os
import time

import numpy as np
import pandas as pd
from scipy import spatial

import skimage as ski

import torch
from PIL import Image

from GSASII_imports import *


# recreating xye export function
# TODO: use numpy or the like to write this faster
def Export_xye(name, data, location, error=True):
    with open(location, "w") as outfile:
        outfile.write("/*\n")
        outfile.write("# {0}\n".format(name))
        outfile.write("*/\n")
        # data = integral['data'][1]
        for i in range(len(data[0])):
            if error:
                outfile.write(
                    "{x}\t{y}\t{e}\n".format(x=data[0][i], y=data[1][i], e=data[2][i])
                )
            else:
                outfile.write("{x}\t{y}\t{e}\n".format(x=data[0][i], y=data[1][i], e=0))


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


# create and save TA[x] maps
# from savemaps.py by Wenqian Xu
def getmaps(
    cache, imctrlname, pathmaps, save=True
):  # fast integration using the same imctrl and mask
    # TA = G2img.Make2ThetaAzimuthMap(imctrls,(0,imctrls['size'][0]),(0,imctrls['size'][1]))    #2-theta array, 2880 according to detector pixel numbers
    imctrls = cache["Image Controls"]
    TA = Make2ThetaAzimuthMap(imctrls, (0, imctrls["size"][0]), (0, imctrls["size"][1]))
    if save:
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
        Qmap = get_Qmap(TA[0], imctrls["wavelength"])
        im = Image.fromarray(Qmap)
        im.save(os.path.splitext(path1)[0] + "_qmap.tif")
        cache["pixelQmap"] = Qmap
    return


def get_Qmap(Tmap, wavelength):
    return 4 * np.pi * np.sin(Tmap / 2 * np.pi / 180) / wavelength


def get_azimbands(azmap, numChansAzim):
    dazim = (360) / numChansAzim
    azimband = np.array(azmap / dazim, dtype=np.int32)
    return azimband


def Qmap(Tmap, wavelength):
    return 4 * np.pi * np.sin(Tmap / 2 * np.pi / 180) / wavelength


def prepare_qmaps(tth_map, pol_map, dist_map, tth_min, tth_max, numChans):
    tth = tth_map.ravel()
    raveled_pol = pol_map.ravel()
    raveled_dist = dist_map.ravel()

    tth_delta = (tth_max - tth_min) / numChans
    tth_list = np.arange(tth_min, tth_max + tth_delta / 2.0, tth_delta)
    tth_val = ((tth_list[1:] + tth_list[:-1]) / 2.0).astype(np.float32)

    tth_idx = np.zeros_like(tth, dtype=np.int32)
    roi_1 = np.zeros_like(tth, dtype=bool)

    for idx, val in enumerate(tth_list[1:]):
        roi_2 = tth < val
        tth_idx[(~roi_1) * roi_2] = idx + 1
        roi_1[:] = roi_2

    tth_idx = torch.from_numpy(tth_idx)
    tth_val = torch.from_numpy(tth_val)
    return tth_idx, tth_val, raveled_pol, raveled_dist, len(tth_val)


def pytorch_integrate(
    image, mask, tth_idx, tth_val, raveled_pol, raveled_dist, tth_size
):
    data = image.ravel()
    data = torch.from_numpy(data)
    mask = mask.ravel()
    mask = ~mask
    mask = torch.from_numpy(mask)
    data = (
        data / raveled_pol * raveled_dist**1.5
    )  # the dist map is squared distance, here it needs to be cubic

    # no masked-array option for torch.bincount or np.bincount
    # val = torch.bincount(tth_idx, weights=data*mask, minlength=tth_size)[1:]
    masked_tth_idx = tth_idx * mask
    val = torch.bincount(masked_tth_idx, weights=data, minlength=tth_size)[1:]
    norm_factor = torch.clamp(torch.bincount(masked_tth_idx)[1:], min=1, max=None)
    val /= norm_factor
    data = torch.vstack([tth_val, val]).numpy().T

    return data


def r_and_phi_hat(image_shape, center):
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
    # calculate distances and x-y-basis angles once for each pixel in footprint
    t0 = time.time()
    if not all([i % 2 == 1 for i in footprint.shape]):
        raise ValueError("Footprint shape must be odd in each direction.")
    central_footprint_point = np.array([i // 2 for i in footprint.shape])
    # print("central footprint point:", central_footprint_point)
    footprint[central_footprint_point[0], central_footprint_point[1]] = 0
    # print("footprint")
    # print(footprint)
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
    # print("central kernel calculation")
    # print("Kernels calculated, getting convolutions")
    r_hat, phi_hat = r_and_phi_hat(image_shape, center)
    t1 = time.time()
    print(
        "Gradient time spent on cache calculations: {0:.2f}s".format(
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


def qwidth_area_classification(
    om,
    Qmap,
    azmap,
    min_arc_area=100,
    Q_max=0.1,
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
        arcs_bool = np.logical_and(
            props_table.loc[:, "area_over_width2"].values > area_Q_shape_min, arcs_bool
        )
        # arcs_bool = np.logical_and(props_table.loc[:,'azim_vs_Q'].values > azim_Q_shape_min, arcs_bool)
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


def split_grad_with_Q(
    image,
    om,
    arc_mask,
    gradient_dict,
    qmap,
    azmap,
    threshold_percentile=0.1,
    return_grad=False,
    return_partials=False,
    interpolate=False,
    predef_mask=None,
    predef_mask_extended=None,
):
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

    # spot_mask = np.in1d(new_mask,spot_clusters)
    Qgrad_arc_mask = np.in1d(labeled_mask, arc_cluster_indices)
    # spot_mask = np.reshape(spot_mask,om.shape)
    Qgrad_arc_mask = np.reshape(Qgrad_arc_mask, om.shape)

    gradient_mask = azim_grad_2 < threshold
    # only look at the areas overlapping potential arcs
    gradient_mask = np.logical_and(gradient_mask, Qgrad_arc_mask)
    # if there aren't any clusters, just skip this part entirely. Maximum will be False if nothing is there.
    if np.max(gradient_mask) > 0:
        labeled_gradients = ski.measure.label(gradient_mask)
        gradient_props = ["label", "centroid", "coords", "area"]
        gradient_props_table = ski.measure.regionprops_table(
            labeled_gradients, properties=gradient_props
        )
        gradient_props_table = pd.DataFrame(gradient_props_table)

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

        cuts = np.sum(gradient_props_table["new_cuts"].values, axis=0).astype(bool)

        # cut section of gradient mask from arc mask
        new_arc_mask = np.logical_and(Qgrad_arc_mask, ~cuts)
        new_spot_mask = np.logical_and(om, ~new_arc_mask)

    else:
        # print("No spots to cut out. Skipping last step.")
        new_arc_mask = Qgrad_arc_mask
        new_spot_mask = np.logical_and(om, ~new_arc_mask)

    # spots stats
    # only arcs have stayed in one table; spots have been scattered across multiple
    # so spots stats table needs to be recreated at this point
    spots_props = ["label", "area", "intensity_mean"]
    labeled_spots = ski.measure.label(new_spot_mask)
    spots_table = ski.measure.regionprops_table(
        labeled_spots, qmap, properties=spots_props
    )
    spots_table = pd.DataFrame(spots_table)

    to_return = [new_spot_mask, new_arc_mask, spots_table]

    if return_grad:
        to_return.append(radial_grad_2)
        to_return.append(azim_grad_2)

    if return_partials:
        to_return.append(Qgrad_arc_mask)

    return to_return


def modulo_range(array, center, range):
    diff = (array - center) % 360
    # range = range.values
    return np.logical_or(diff < range, diff > (360 - range))


def current_splitting_method(
    image,
    om,
    qmap,
    azmap,
    gradient_dict,
    threshold_percentile=0.1,
    return_steps=False,
    interpolate=False,
    predef_mask=None,
    predef_mask_extended=None,
):
    # t0 = time.time()
    # base_spot, base_arc = qwidth_area_classification(om, qmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_max = 0.00001)
    # base_spot, base_arc = qwidth_area_classification(om, qmap, azmap, min_arc_area=100, max_width=0.2, compare_shape=True, shape_min = 3500)
    base_spot, base_arc = qwidth_area_classification(
        om, qmap, azmap, min_arc_area=100, Q_max=0.08, azim_min=3.5, compare_shape=True
    )
    # t1 = time.time()
    # print("Time to do initial q width / area classification: {0:.2f}s".format(t1-t0))
    # spot_mask, arc_mask = split_grad(image,om,base_arc,gradient_dict,threshold_percentile=threshold_percentile)
    if return_steps:
        spot_mask, arc_mask, spots_table, radial_grad_2, azim_grad_2, qgrad_arc_mask = (
            split_grad_with_Q(
                image,
                om,
                base_arc,
                gradient_dict,
                qmap,
                azmap,
                threshold_percentile=threshold_percentile,
                return_partials=True,
                return_grad=True,
                interpolate=interpolate,
                predef_mask=predef_mask,
                predef_mask_extended=predef_mask_extended,
            )
        )
    else:
        spot_mask, arc_mask, spots_table = split_grad_with_Q(
            image,
            om,
            base_arc,
            gradient_dict,
            qmap,
            azmap,
            threshold_percentile=threshold_percentile,
            interpolate=interpolate,
            predef_mask=predef_mask,
            predef_mask_extended=predef_mask_extended,
        )

    to_return = [spot_mask, arc_mask, spots_table]
    if return_steps:
        to_return.append(base_arc)
        to_return.append(qgrad_arc_mask)
        to_return.append(azim_grad_2)
        to_return.append(radial_grad_2)
    return to_return


def run_cache(filename, input_directory, output_directory, imctrlname, blkSize, imgmaskname=None, flatfield=None, ):
    """
    Creates and returns the initial cache.

    Parameters
    ----------
    filename: str
        Name of the image file to run over

    input_directory: str
        Path to the directory the image files sit in

    output_directory: st
        Path to the directory to place output files such as integrals
    
    imctrlname: str
        Name of the image control file

    blkSize: int
        Size of the integration blocks

    imgmaskname: str
        Name of the predefined image mask

    flatfield: str
        Name of the flatfield image
    """
    cache = {}
    image_dict = read_image(filename)
    # img.loadControls(imctrlname)   # set controls/calibrations/masks
    with open(imctrlname, "r") as imctrlfile:
        lines = imctrlfile.readlines()
        LoadControls(lines, image_dict["Image Controls"])
    # cache['Image Controls'] = img.getControls() # save controls & masks contents for quick reload
    # self.cache['image'] = tf.imread(self.filename)
    cache["image"] = load_image(filename)

    predef_mask = {}
    if (imgmaskname is not None) and (imgmaskname != ""):
        # img.loadMasks(imgmaskname)
        suffix = imgmaskname.split(".")[1]
        if suffix == "immask":
            readMasks(imgmaskname, image_dict["Masks"], False)
        elif suffix == "tif":
            print(imgmaskname)
            predef_mask = read_image(imgmaskname)
    else:
        predef_mask["image"] = np.zeros_like(image_dict["image"], dtype=bool)
    cache["predef_mask"] = predef_mask

    flatfield_image = np.ones_like(cache["image"])
    if (flatfield is not None) and (flatfield != ""):
        # flatfield_image = tf.imread(self.flatfield)
        flatfield_image = load_image(flatfield)
    cache["flatfield"] = flatfield_image
    
    # imctrlname = imctrlname.split("\\")[-1].split('/')[-1]
    # path1 =  os.path.join(pathmaps,imctrlname)
    # im = Image.fromarray(TA[0])
    # im.save(os.path.splitext(path1)[0] + '_2thetamap.tif')
    imsave = Image.fromarray(predef_mask["image"])
    imsave.save(
        os.path.join(
            output_directory,
            "maps",
            os.path.splitext(os.path.split(imctrlname)[1])[0] + "_predef.tif"
        )
    )
    imsave = Image.fromarray(flatfield_image)
    imsave.save(
        os.path.join(
            output_directory,
            "maps",
            os.path.splitext(os.path.split(imctrlname)[1])[0] + "_flatfield.tif"
        )
    )
    cache["Image Controls"] = image_dict["Image Controls"]
    # TODO: Look at image size?
    # img.setControl('pixelSize',[150.0,150.0])
    image_dict["Image Controls"]["pixelSize"] = [150.0, 150.0]
    cache["Image Controls"]["pixelSize"] = [150.0, 150.0]
    # cache['Masks'] = img.getMasks()
    # self.cache['Masks'] = image_dict['Masks']
    # cache['intMaskMap'] = img.IntMaskMap() # calc mask & TA arrays to save for integrations
    # for k,v in img_copy['Image Controls'].items():
    #    print(k)
    # for k in img.data['Image Controls'].keys():
    #    if k not in img_copy['Image Controls'].keys():
    #        print(k, img.data['Image Controls'][k])
    # Missing 5: size, samplechangerpos, det2theta, ImageTag, formatName
    # [2880,2880] None 0.0 None GSAS-II known TIF image
    # only size seems to be holding anything meaningful at this time, though det2theta and samplechangerpos could hold something later
    
    # self.cache["intMaskMap"] = MakeUseMask(
    #     image_dict["Image Controls"],image_dict["Masks"], blkSize=self.blkSize
    # )
    # cache['intTAmap'] = img.IntThetaAzMap()
    cache["intTAmap"] = MakeUseTA(image_dict["Image Controls"], blkSize)
    # cache['FrameMask'] = img.MaskFrameMask() # calc Frame mask & T array to save for Pixel masking
    cache["FrameMask"] = MaskFrameMask(image_dict)
    # cache['maskTmap'] = img.MaskThetaMap()
    cache["maskTmap"] = Make2ThetaAzimuthMap(
        image_dict["Image Controls"],
        (0, image_dict["Image Controls"]["size"][0]),
        (0, image_dict["Image Controls"]["size"][1])
    )[0]
    getmaps(cache, imctrlname, os.path.join(output_directory, "maps"))
    # 2th fairly linear along center; calc 2th - pixelsize conversion
    center = cache["Image Controls"]["center"]
    center[0] = center[0] * 1000.0 / cache["Image Controls"]["pixelSize"][0]
    center[1] = center[1] * 1000.0 / cache["Image Controls"]["pixelSize"][1]
    cache["center"] = center
    image_dict["center"] = center
    # self.cache['d2th'] = (self.cache['pixelTAmap'][int(center[1]),0] - self.cache['pixelTAmap'][int(center[1]),99])/100
    cache["esdMul"] = 3
    numChansAzim = 360
    cache["azimband"] = get_azimbands(cache["pixelAzmap"], numChansAzim)

    # pytorch integration
    (
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    ) = prepare_qmaps(
        cache["pixelTAmap"],
        cache["polscalemap"],
        cache["pixelsampledistmap"],
        cache["Image Controls"]["IOtth"][0],
        cache["Image Controls"]["IOtth"][1],
        cache["Image Controls"]["outChannels"],
    )

    # comparisons
    # self.cache['Previous image'] = tf.imread(self.filename)
    # self.cache['First image'] = tf.imread(self.filename)
    cache["First image"] = load_image(filename)

    # gradient info
    cache["gradient"] = gradient_cache(
        predef_mask["image"].shape, center, np.ones((3, 3), dtype=np.uint)
    )

    cache["image_dict"] = image_dict

    return cache

def run_iteration(filename, input_directory, output_directory, name, number, cache, ext, closing_method = "binary_closing", return_steps = False):
    """
    Runs over each file, outputting masks and integral files.

    Parameters
    ----------
    filename: str
        Name of the file to run over

    input_directory: str
        Path to the directory the images are located

    output_directory: str
        Path to the directory to place output files such as integrals

    name: str

    number:

    cache: dict
        Output from run_cache()

    ext: str

    closing_method: str
        Method for removing small holes in the outlier mask.
        Default is binary_closing.

    return_steps: bool
        Save intermediate masks and gradients.
        Default is False.

    """
    image_dict = cache["image_dict"]
    # image_dict['image'] = tf.imread(self.filename)
    image_dict["image"] = load_image(filename)
    # add the correction in now
    image_dict["image"] = flatfield_correct(
        image_dict["image"], cache["flatfield"]
    )
    image_dict["corrected_image"] = None
    nonpositive_mask = ~nonzeromask(image_dict["image"], mask_negative=True)
    imsave = Image.fromarray(nonpositive_mask)
    imsave.save(
        os.path.join(
            output_directory,
            "masks",
            name + "-" + number + "_nonpositive.tif"
        )
    )
    predef_and_nonpositive = np.logical_or(
        nonpositive_mask, cache["predef_mask"]["image"]
    )
    predef_mask_extended = ski.morphology.binary_dilation(
        predef_and_nonpositive, footprint=ski.morphology.square(7)
    )  # extend out by three pixels; use for determining whether something is nearby
    frame_and_predef = np.logical_or(
        predef_and_nonpositive, cache["FrameMask"]
    )
    GeneratePixelMask(
        image_dict,
        esdMul=cache["esdMul"],
        FrameMask=frame_and_predef,
        ThetaMap=cache["maskTmap"],
    )
    # outlier_mask = img.data['Masks']['SpotMask']['spotMask']
    outlier_mask = image_dict["Masks"]["SpotMask"]["spotMask"]
    imsave = Image.fromarray(outlier_mask)
    imsave.save(
        os.path.join(
            output_directory,
            "masks",
            name + "-" + number + "_om.tif"
        )
    )
    # close holes
    if closing_method == "binary_closing":
        closed_mask = ski.morphology.binary_closing(
            outlier_mask, footprint=ski.morphology.square(3)
        )
        imsave = Image.fromarray(closed_mask)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_closedmask.tif"
            )
        )

    elif closing_method == "remove_small":
        closed_mask = ski.morphology.remove_small_holes(outlier_mask, 6)
        imsave = Image.fromarray(closed_mask)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_closedmask.tif"
            )
        )
    elif (closing_method == None) or (closing_method == ""):
        closed_mask = outlier_mask
    else:
        print("Unrecognized closing method: Using none")
        closed_mask = outlier_mask

    return_steps = False
    if return_steps:
        (
            split_spots,
            split_arcs,
            spots_table,
            base_arc,
            qgrad_arc,
            azim_grad_2,
            radial_grad_2,
        ) = current_splitting_method(
            image_dict["image"],
            closed_mask,
            cache["pixelQmap"],
            cache["pixelAzmap"],
            cache["gradient"],
            return_steps=return_steps,
            interpolate=False,
            predef_mask=nonpositive_mask,
            predef_mask_extended=predef_mask_extended,
        )
        imsave = Image.fromarray(split_spots)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_spots.tif"
            )
        )
        imsave = Image.fromarray(split_arcs)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_arcs.tif"
            )
        )
        imsave = Image.fromarray(base_arc)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_qwidth_arc.tif"
            )
        )
        imsave = Image.fromarray(qgrad_arc)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_qgrad_arc.tif"
            )
        )
        imsave = Image.fromarray(azim_grad_2)
        imsave.save(
            os.path.join(
                output_directory,
                "grads",
                name + "-" + number + "_azim_grad_2.tif"
            )
        )
        imsave = Image.fromarray(radial_grad_2)
        imsave.save(
            os.path.join(
                output_directory,
                "grads",
                name + "-" + number + "_radial_grad_2.tif"
            )
        )
    else:
        (
            split_spots,
            split_arcs,
            spots_table,
        ) = current_splitting_method(
            image_dict["image"],
            closed_mask,
            cache["pixelQmap"],
            cache["pixelAzmap"],
            cache["gradient"],
            return_steps=return_steps,
            interpolate=False,
            predef_mask=nonpositive_mask,
            predef_mask_extended=predef_mask_extended
        )
        imsave = Image.fromarray(split_spots)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_spots.tif"
            )
        )
        imsave = Image.fromarray(split_arcs)
        imsave.save(
            os.path.join(
                output_directory,
                "masks",
                name + "-" + number + "_arcs.tif"
            )
        )



    # integrate
    hist_base = pytorch_integrate(
        image_dict["image"],
        frame_and_predef,
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    )
    hist_closed = pytorch_integrate(
        image_dict["image"],
        np.logical_or(closed_mask, frame_and_predef),
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    )
    hist_closedspotsmasked = pytorch_integrate(
        image_dict["image"],
        np.logical_or(split_spots, frame_and_predef),
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    )
    hist_closedarcsmasked = pytorch_integrate(
        image_dict["image"],
        np.logical_or(split_arcs, frame_and_predef),
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    )
    # save integrals
    integral_file_base = os.path.join(
        output_directory,
        "integrals",
        name + "-" + number
    )
    Export_xye(
        name + "-" + number + "_base",
        hist_base.T,
        integral_file_base + "_base.xye",
        error=False,
    )
    Export_xye(
        name + "-" + number + "_closed",
        hist_closed.T,
        integral_file_base + "_closed.xye",
        error=False,
    )
    Export_xye(
        name + "-" + number + "_closedspotsmasked",
        hist_closedspotsmasked.T,
        integral_file_base + "_closedspotsmasked.xye",
        error=False,
    )
    Export_xye(
        name + "-" + number + "_closedarcsmasked",
        hist_closedarcsmasked.T,
        integral_file_base + "_closedarcsmasked.xye",
        error=False,
    )
    
    # stats
    stats_prefix = os.path.join(output_directory, "stats", name)
    # spots stats
    spots_table.to_csv(stats_prefix + "-" + number + "_spots_stats.csv")
    # ~ 950 KB for table
    # 2d histogram: area, Q position
    # ~7800 KB for 1000x1000 bin histogram
    # 81 KB for 100x100 bin histogram
    hist, x_edges, y_edges = np.histogram2d(
        spots_table["area"].values, spots_table["intensity_mean"].values, 100
    )
    with open(
        stats_prefix + "-" + number + "_spots_hist.npy", "wb"
    ) as outfile:
        np.save(outfile, hist)
        np.save(outfile, x_edges)
        np.save(outfile, y_edges)

    # Calculate comparisons between images
    # Find and read in previous image given current image number
    prev_number = ""
    number_int_prev = int(number) - 1
    if number_int_prev < 0:
        # first image (00000) will have no previous image; just compare to self
        prev_number = number
    else:
        # turn int back to '00001' format, padded to 5 digits
        prev_number = f"{number_int_prev:05}"
    try:
        previous_image = ski.io.imread(
            os.path.join(input_directory, name + "-" + prev_number + ext)
        ).astype(np.float32)
    except:
        print("Cannot find previous image for cosine similarity; using current instead.")
        previous_image = image_dict["image"].astype(np.float32)
    csim_f = 1 - spatial.distance.cosine(
        np.array(image_dict["image"], dtype=np.float32).ravel(),
        cache["First image"].ravel(),
    )
    csim_p = 1 - spatial.distance.cosine(
        np.array(image_dict["image"], dtype=np.float32).ravel(),
        previous_image.ravel(),
    )
    with open(stats_prefix + "-" + number + "_csim.txt", "w") as outfile:
        outfile.write(
            "{first:0.4f}\t{prev:0.4f}\n".format(first=csim_f, prev=csim_p)
        )

    # Also not safe for multiprocessing. Need to grab number-1 when it exists
    cache["Previous image"] = image_dict["image"]

    