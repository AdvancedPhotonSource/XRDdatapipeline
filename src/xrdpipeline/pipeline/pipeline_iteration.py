"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the main routines used in the analysis pipeline.
"""

import os, sys
import time

import numpy as np
from scipy import spatial

import skimage as ski

import torch
from PIL import Image

import argparse
import re
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
mid_dir = os.path.split(script_dir)[0]
if mid_dir not in sys.path:
    sys.path.append(mid_dir)

from pipeline.classification import current_splitting_method
from general.corrections_and_maps import *
from general.file_name_definitions import add_output_subdirectory, find_name_number, ImageNumberStyle
from general.GSASII_imports import *


# recreating xye export function
# TODO: use numpy or the like to write this faster
def Export_xye(name, data, location, error=True):
    """
    Export a set of integral data or similar to an xye file (x value, y value, error)

    :param name: Name of the image
    :param data: Array of data to write to the file
    :param location: Location of the file to write, without an extension
    :param error: Boolean of whether there is an error column in the data
    """
    location += ".xye"
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


def Export_chi(name, data, location):
    """
    Export a set of integral data or the like to a .chi file.

    :param name: Name of the image
    :param data: Array of data to write to the file
    :param location: Location of the file to write, without an extension
    """
    location += ".chi"
    data_len = len(data[0])
    with open(location,"w") as outfile:
        outfile.write(f"{name} Azm= 0.00\n")
        outfile.write("2-Theta Angle (Degrees)\nIntensity\n")
        outfile.write(f"       {data_len}\n")
        for i in range(data_len):
            outfile.write(f" {data[0][i]:.7e}   {data[1][i]:.7e}\n")


def pytorch_data_setup(image, raveled_pol, raveled_dist):
    """
    Prepare the image data to be integrated using pytorch methods

    :param image: 2d image data
    :param raveled_pol: 1d raveling of polarization data for each pixel
    :param raveled_dist: 1d raveling of the distance map for each pixel
    """
    data = image.ravel()
    data = torch.from_numpy(data)
    data = (
        data / raveled_pol * raveled_dist
    )
    return data

def pytorch_data_setup_2(image):
    """
    Prepare the image data to be integrated using pytorch methods

    The image has been correted for polarization and distance already, so this just ravels and turns to torch.

    :param image: 2d image data
    """
    data = image.ravel()
    data = torch.from_numpy(data)
    return data


def pytorch_integrate(
    data, mask, tth_idx, tth_val, tth_size
):
    """
    Fast integration method.

    :param data: 1d ravel of prepared image data using pytorch_data_setup()
    :param mask: 2d array of pixels to mask
    :param tth_idx: 1d list of central bin 2theta values
    :param tth_val: 1d ravel of 2theta values for each pixel
    :param tth_size: Number of bins
    """
    mask = mask.ravel()
    mask = ~mask
    mask = torch.from_numpy(mask)

    # no masked-array option for torch.bincount or np.bincount
    # val = torch.bincount(tth_idx, weights=data*mask, minlength=tth_size)[1:]
    masked_tth_idx = tth_idx * mask
    val = torch.bincount(masked_tth_idx, weights=data, minlength=tth_size)[1:]
    norm_factor = torch.clamp(torch.bincount(masked_tth_idx)[1:], min=1, max=None)
    val /= norm_factor
    data = torch.vstack([tth_val, val]).numpy().T

    return data


def run_iteration(
        filename,
        input_directory,
        output_directory,
        name,
        ext,
        cache_location = None,
        cache = None,
        closing_method = "binary_closing",
        calc_outlier = True,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        min_cluster_area = 3,
        min_arc_area = 100,
        min_azim_width = 0,
        max_q_width = 0.1,
        spot_threshold_percentile = 0.1,
        arc_threshold_percentile = 10,
        calc_spot_stats = True,
        calc_grad_spottiness = False,
        calc_azim_Qs = True,
        use_radial_grad = True,
        use_azim_grad = True,
        calc_csim = True,
        csim_first_index = 0,
        n_mask_bins = 1000,
        timing = None,
        timing_names = None,
    ):
    """
    Runs over each file, outputting masks and integral files.

    :param filename: Name of the file to run over
    :param input_directory: Path to the directory the images are located
    :param output_directory: Path to the directory to place output files such as integrals
    :param name: Name of the dataset
    :param ext: Extension of the file name
    :param cache_location: Location of the cache file saved to disk. One of this or the cache parameter is required.
    :param cache: Dictionary output from run_cache(). One of this or the cache_location parameter is required.
    :param closing_method: Method for removing small holes in the outlier mask. Default is binary_closing.
    :param calc_outlier: Whether to calculate an outlier mask. Default is True.
    :param calc_splitting: Whether to split the outlier mask into spot-tagged and texture-tagged clusters. Default is True.
    :param azim_Q_shape_min: Ratio of the azimuthal to Q widths to use for an early cut when splitting spots from textures. Default is 100.
    :param min_cluster_area: Clusters must be larger than this pixel area to be classified as spot or arc
    :param min_arc_area: Minimum cluster area in pixels to be determined a texture arc
    :param min_azim_width: Minimum azimuthal width in degrees for a cluster to be determined a texture arc
    :param max_q_width: Maximum Q width for a cluster to be determined a texture arc
    :param spot_threshold_percentile: Percentile of the radial second derivative intensities to use as a threshold to find spots in texture arcs
    :param arc_threshold_percentile: Percentile of the radial second derivative intensities to use as a threshold to determine if a cluster is on a powder ring
    :param calc_spot_stats: Whether to calculate spottiness statistics based on the stats collected on spot-tagged clusters. Does not add a lot of computation time. Default is True.
    :param calc_grad_spottiness: Whether to use information on the second azimuthal and Q derivatives of the image to calculate spottiness statistics. Adds significant computation time. Default is False.
    :param calc_azim_Qs: Calculate and save azimuth / Q spans for all clusters
    :param use_radial_grad: Use radial second derivative information to check if clusters are on a powder arc
    :param use_azim_grad: Use azimuthal second derivative information to cut spots from texture arc candidates
    :param calc_csim: Whether to calculate cosine similarity. Default is True.
    :param csim_first_index: Cosine similarity is calculated in comparison to the previous image in the set and the first image in the dataset. This is the index marking the first image. Default is 0.
    :param n_mask_bins: Number of 2theta bins to use for outlier masking. Default is 1000.
    :param timing: Timing information.
    :param timing_names: Names to print for each timing checkpoint. These will be generated if None is passed. Default is None.
    """
    # Sanity checks to adjust defaults
    if not calc_outlier:
        calc_splitting = False
    if not calc_splitting:
        calc_spot_stats = False
        calc_grad_spottiness = False
        calc_azim_Qs = False
        use_radial_grad = False
        use_azim_grad = False
    if not (use_azim_grad or use_radial_grad):
        calc_grad_spottiness = False

    output_directory = add_output_subdirectory(output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    newdirs = ["maps", "masks", "integrals", "stats", "logs"]
    for newdir in newdirs:
        path = os.path.join(output_directory, newdir)
        if not os.path.exists(path):
            os.mkdir(path)

    short_name, number, style = find_name_number(name)

    if timing is not None:
        timing_0 = time.time()
        local_times = []
    if cache is None and cache_location is None:
        print("Pipeline requires one of cache or cache location.")
        return
    if cache is None and cache_location is not None:
        cache = np.load(cache_location,allow_pickle=True).item()
    if timing is not None:
        timing_1 = time.time()
        local_times.append(timing_1-timing_0)
        timing_name = "Load cache"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        timing_0 = time.time()
    image_dict = cache["image_dict"]
    # image_dict['image'] = tf.imread(self.filename)
    image_dict["image"] = load_image(filename)
    # add the correction in now
    if cache["flatfield"] is not None:
        image_dict["image"] = flatfield_correct(
            image_dict["image"], cache["flatfield"]
        )
        imsave = Image.fromarray(image_dict["image"])
        imsave.save(
            os.path.join(
                output_directory,
                "flatfield",
                name + "_flatfield_correct.tif"
            )
        )

    if cache["polscalemap"] is not None:
        image_dict["image"] = pol_correct(
            image_dict["image"], cache["polscalemap"]
        )
    else:
        print("polscalemap not found in cache; skipping polarization correction.")
    if cache["pixelsampledistmap"] is not None:
        image_dict["image"] = dist_correct(
            image_dict["image"], cache["pixelsampledistmap"]
        )
    else:
        print("pixelsampledistmap not found in cache; skipping distance correction.")
    #image_dict["corrected_image"] = None
    image_dict["corrected_image"] = image_dict["image"].astype(np.int32)
    nonpositive_mask = ~nonzeromask(image_dict["image"], mask_negative=True)
    predef_and_nonpositive = np.logical_or(
        nonpositive_mask, cache["predef_mask"]["image"]
    )
    imsave = Image.fromarray(predef_and_nonpositive)
    imsave.save(
        os.path.join(
            output_directory,
            "masks",
            name + "_base.tif"
        )
    )
    # predef_mask_extended = ski.morphology.binary_dilation(
    #     predef_and_nonpositive, footprint=ski.morphology.square(7)
    # )  # extend out by three pixels; use for determining whether something is nearby
    if timing is not None:
        timing_1 = time.time()
        local_times.append(timing_1-timing_0)
        timing_name = "Initial corrections"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        timing_0 = time.time()
    if calc_outlier:
        esdMul = cache["esdMul"]
        GeneratePixelMask(
            image_dict,
            esdMul=esdMul,
            FrameMask=predef_and_nonpositive,
            ThetaMap=cache["pixelTAmap"],
            numChans=n_mask_bins,
        )
        # outlier_mask = img.data['Masks']['SpotMask']['spotMask']
        outlier_mask = image_dict["Masks"]["SpotMask"]["spotMask"]
        if timing is not None:
            timing_1 = time.time()
            local_times.append(timing_1-timing_0)
            timing_name = "Outlier mask"
            if timing_name not in timing_names:
                timing_names.append(timing_name)
            timing_0 = time.time()
        # close holes
        if closing_method == "binary_closing":
            t0 = time.time()
            closed_mask = ski.morphology.binary_closing(
                outlier_mask, footprint=ski.morphology.square(3)
            )
            imsave = Image.fromarray(closed_mask)
            imsave.save(
                os.path.join(
                    output_directory,
                    "masks",
                    name + "_outliermask.tif"
                )
            )
            t1 = time.time()
            # print(f"Binary closing time: {t1-t0}")

        elif closing_method == "remove_small":
            closed_mask = ski.morphology.remove_small_holes(outlier_mask, 6)
            imsave = Image.fromarray(closed_mask)
            imsave.save(
                os.path.join(
                    output_directory,
                    "masks",
                    name + "_outliermask.tif"
                )
            )
        elif (closing_method == None) or (closing_method == ""):
            closed_mask = outlier_mask
        else:
            print("Unrecognized closing method: Using none")
            closed_mask = outlier_mask
        if timing is not None:
            timing_1 = time.time()
            local_times.append(timing_1-timing_0)
            timing_name = "Binary closing"
            if timing_name not in timing_names:
                timing_names.append(timing_name)
            timing_0 = time.time()

        if calc_splitting:
            if timing is not None:
                returned_items = current_splitting_method(
                    image_dict["image"].copy(),
                    closed_mask,
                    cache["pixelQmap"],
                    cache["pixelAzmap"],
                    cache["gradient"],
                    cache["Qbins"],
                    spot_threshold_percentile=spot_threshold_percentile,
                    arc_threshold_percentile=arc_threshold_percentile,
                    calc_spot_stats = calc_spot_stats,
                    calc_grad_spottiness=calc_grad_spottiness,
                    calc_azim_Qs=calc_azim_Qs,
                    use_radial_grad=use_radial_grad,
                    use_azim_grad=use_azim_grad,
                    azim_Q_shape_min=azim_Q_shape_min,
                    predef_mask=nonpositive_mask,
                    min_cluster_area=min_cluster_area,
                    min_arc_area=min_arc_area,
                    min_azim_width=min_azim_width,
                    max_Q_width=max_q_width,
                    timing = local_times,
                    timing_names = timing_names,
                )
            else:
                returned_items = current_splitting_method(
                    image_dict["image"].copy(),
                    closed_mask,
                    cache["pixelQmap"],
                    cache["pixelAzmap"],
                    cache["gradient"],
                    cache["Qbins"],
                    spot_threshold_percentile=spot_threshold_percentile,
                    arc_threshold_percentile=arc_threshold_percentile,
                    calc_spot_stats = calc_spot_stats,
                    calc_grad_spottiness=calc_grad_spottiness,
                    calc_azim_Qs=calc_azim_Qs,
                    use_radial_grad=use_radial_grad,
                    use_azim_grad=use_azim_grad,
                    azim_Q_shape_min=azim_Q_shape_min,
                    predef_mask=nonpositive_mask,
                    min_cluster_area=min_cluster_area,
                    min_arc_area=min_arc_area,
                    min_azim_width=min_azim_width,
                    max_Q_width=max_q_width,
                    timing = None,
                    timing_names = None,
                )
            returned_items = list(returned_items)
            split_spots = returned_items.pop(0)
            split_arcs = returned_items.pop(0)
            if calc_grad_spottiness:
                spots_table_grad = returned_items.pop(0)
            if calc_spot_stats:
                spots_table_df = returned_items.pop(0)
            if calc_azim_Qs:
                azim_vs_Qs = returned_items.pop(0)

            imsave = Image.fromarray(split_spots)
            imsave.save(
                os.path.join(
                    output_directory,
                    "masks",
                    name + "_spots.tif"
                )
            )
            imsave = Image.fromarray(split_arcs)
            imsave.save(
                os.path.join(
                    output_directory,
                    "masks",
                    name + "_arcs.tif"
                )
            )
            if timing is not None:
                timing_1 = time.time()
                local_times.append(timing_1-timing_0)
                timing_name = "Total mask splitting"
                if timing_name not in timing_names:
                    timing_names.append(timing_name)
                timing_0 = time.time()


    # prep data
    #corrected_image_data = pytorch_data_setup(image_dict["image"], cache["raveled_pol"], cache["raveled_dist"])
    corrected_image_data = pytorch_data_setup_2(image_dict["corrected_image"])
    # integrate
    base_mask = predef_and_nonpositive | cache["AzimMask"]
    hist_base = pytorch_integrate(
        corrected_image_data,
        base_mask,
        cache["tth_idx"],
        cache["tth_val"],
        cache["tth_size"],
    )
    if calc_outlier:
        hist_closed = pytorch_integrate(
            corrected_image_data,
            np.logical_or(closed_mask, base_mask),
            cache["tth_idx"],
            cache["tth_val"],
            cache["tth_size"],
        )
        if calc_splitting:
            hist_closedspotsmasked = pytorch_integrate(
                corrected_image_data,
                np.logical_or(split_spots, base_mask),
                cache["tth_idx"],
                cache["tth_val"],
                cache["tth_size"],
            )
            hist_closedarcsmasked = pytorch_integrate(
                corrected_image_data,
                np.logical_or(split_arcs, base_mask),
                cache["tth_idx"],
                cache["tth_val"],
                cache["tth_size"],
            )
    # save integrals
    integral_file_base = os.path.join(
        output_directory,
        "integrals",
        name
    )
    Export_chi(
        name + ".tif",
        hist_base.T,
        integral_file_base + "_base",
        # error=False,
    )
    if calc_outlier:
        Export_chi(
            name + ".tif",
            hist_closed.T,
            integral_file_base + "_om",
            # error=False,
        )
        if calc_splitting:
            Export_chi(
                name + ".tif",
                hist_closedspotsmasked.T,
                integral_file_base + "_spotsmasked",
                # error=False,
            )
            Export_chi(
                name + ".tif",
                hist_closedarcsmasked.T,
                integral_file_base + "_arcsmasked",
                # error=False,
            )
    if timing is not None:
        timing_1 = time.time()
        local_times.append(timing_1-timing_0)
        timing_name = "Integrations"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        timing_0 = time.time()
    
    stats_prefix = os.path.join(output_directory, "stats", name)
    if calc_outlier and calc_splitting:
        # spottiness
        if calc_azim_Qs:
            azim_vs_Qs.to_csv(stats_prefix + "_azim_vs_Qs.csv")
        if calc_spot_stats or calc_grad_spottiness:
            if calc_spot_stats:
                spots_table_df.to_csv(stats_prefix + "_spots_stats_df.csv")
            if calc_grad_spottiness:
                spots_table_grad.to_csv(stats_prefix + "_spots_stats_grad.csv")
            qbins_filename = os.path.join(output_directory, "stats","qbinedges.npy")
            if not os.path.exists(qbins_filename):
                with open(qbins_filename, "wb") as outfile:
                    np.save(outfile, cache["QbinEdges"])
        if timing is not None:
            timing_1 = time.time()
            local_times.append(timing_1-timing_0)
            timing_name = "Save stats"
            if timing_name not in timing_names:
                timing_names.append(timing_name)
            timing_0 = time.time()

    # Calculate comparisons between images
    # Find and read in previous image given current image number
    if style != ImageNumberStyle.NoNumber:
        prev_number = ""
    if calc_csim:
        number_int_prev = int(number) - 1
        if number_int_prev < 0:
            # first image (00000) will have no previous image; just compare to self
            prev_number = number
        else:
            # turn int back to '00001' format, padded to 5 digits
            if style == ImageNumberStyle.Default:
                prev_number = f"{number_int_prev:05}"
                first_index_str = f"{csim_first_index:0>5}"
            elif style == ImageNumberStyle.NumberOnly:
                prev_number = f"{number_int_prev}"
                first_index_str = f"{csim_first_index}"
        try:
            num_splits = ["-", "_", ""]
            for split in num_splits:
                previous_image_name = os.path.join(input_directory, short_name + split + prev_number + ext)
                if os.path.exists(previous_image_name):
                    previous_image = ski.io.imread(
                        previous_image_name
                    ).astype(np.float32)
                    break
            if previous_image is None:
                print("Cannot find previous image for cosine similarity; using current instead.")
                previous_image = image_dict["image"].astype(np.float32)
        except:
            print("Exception finding previous image for cosine similarity; using current instead.")
            previous_image = image_dict["image"].astype(np.float32)
    
        try:
            first_image = None
            for split in num_splits:
                first_image_name = os.path.join(input_directory, short_name + split + first_index_str + ext)
                if os.path.exists(first_image_name):
                    first_image = ski.io.imread(
                        first_image_name
                    ).astype(np.float32)
                    break
            if (first_image is None) and (style == ImageNumberStyle.Default):
                for split in num_splits:
                    first_image_name_ex0 = os.path.join(
                        input_directory, short_name + split + "00000-" + first_index_str + ext
                    )
                    first_image_name_cut = os.path.join(
                        input_directory, short_name[:-6] + split + first_index_str + ext
                    )
                    if os.path.exists(first_image_name_ex0):
                        first_image = ski.io.imread(
                            first_image_name
                        ).astype(np.float32)
                        break
                    elif os.path.exists(first_image_name_cut):
                        first_image = ski.io.imread(
                            first_image_name_cut
                        ).astype(np.float32)
                        break
            if first_image is None:
                print("Cannot find first image for cosine similarity; using current instead.")
                first_image = image_dict["image"].astype(np.float32)
        except:
            print("Exception finding first image for cosine similarity; using current instead.")
            first_image = image_dict["image"].astype(np.float32)
    
        csim_f = 1 - spatial.distance.cosine(
            np.array(image_dict["image"], dtype=np.float32).ravel(),
            first_image.ravel(),
        )
        csim_p = 1 - spatial.distance.cosine(
            np.array(image_dict["image"], dtype=np.float32).ravel(),
            previous_image.ravel(),
        )
        with open(stats_prefix + "_csim.txt", "w") as outfile:
            outfile.write(
                "{first:0.9f}\t{prev:0.9f}\n".format(first=csim_f, prev=csim_p)
            )
    
        if timing is not None:
            timing_1 = time.time()
            local_times.append(timing_1-timing_0)
            timing_name = "Cosine Similarity"
            if timing_name not in timing_names:
                timing_names.append(timing_name)
    
    if timing is not None:
        timing.append(local_times)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True, help="Input file")
    parser.add_argument("-i", "--input_directory", required=True, help="Location of the input image directory")
    parser.add_argument("-o", "--output_directory", required=True, help="Location to place the output files from this pipeline")
    parser.add_argument("-c", "--cache_location", required=True, help="Cache location")
    parser.add_argument("--outlier_option", choices=["splitting", "outlier_only", "none"], default="splitting", help="Choose whether to perform no outlier masking, outlier masking only, or outlier masking with spot/texture splitting.")
    parser.add_argument("--min_cluster_area", type=int, default=3, help="Clusters must be larger than this pixel area to be classified as spot or arc")
    parser.add_argument("--min_arc_area", type=int, default=100, help="Minimum cluster area in pixels to be determined a texture arc")
    parser.add_argument("--min_azim_width", type=float, default=0, help="Minimum azimuthal width in degrees for a cluster to be determined a texture arc")
    parser.add_argument("--max_q_width", type=float, default=0.1, help="Maximum Q width for a cluster to be determined a texture arc")
    parser.add_argument("-a", "--azim_Q_ratio", type=int, default=100, help="Azimuthal to Q width ratio used for classifying spots. Default is 100.")
    parser.add_argument("--skip_radial_grad", dest="use_radial_grad", action='store_false', help="Skip use of radial second derivative information to check if clusters are on a powder arc")
    parser.add_argument("--skip_azim_grad", dest="use_azim_grad", action='store_false', help="Skip use of azimuthal second derivative information to cut spots from texture arc candidates")
    parser.add_argument("--spot_threshold_percentile", type=float, default=0.1, help="Percentile of the radial second derivative intensities to use as a threshold to find spots in texture arcs")
    parser.add_argument("--arc_threshold_percentile", type=float, default=10, help="Percentile of the radial second derivative intensities to use as a threshold to determine if a cluster is on a powder ring")
    parser.add_argument("--spottiness_option", choices=["spot_and_gradient","spot_area_only","none"], default="spot_area_only", help="Choose whether to perform spottiness statistics calculations.")
    parser.add_argument("--skip_calc_azim_Qs", dest="calc_azim_Qs", action='store_false', help="Do not calculate and save azimuth / Q spans for all clusters")
    parser.add_argument("--skip_csim", dest="calc_csim", action='store_false', help="Skip cosine similarity calculation")
    parser.add_argument("--csim_first_index", type=int, default=0, help="Numerical index for the file which should be considered first when calculating cosine similarity.")
    parser.add_argument("--files_must_include", help="Process only files in the directory which include the provided string in their name.")
    parser.add_argument("--files_must_exclude", help="Exclude files in the directory which have the provided string in their name.")
    args = parser.parse_args()

    calc_outlier = True
    calc_splitting = True
    if args.outlier_option == "outlier_only":
        calc_splitting = False
    elif args.outlier_option == "none":
        calc_splitting = False
        calc_outlier = False
    calc_spot_stats = True
    calc_grad_spottiness = False
    if args.spottiness_option == "spot_and_gradient":
        calc_grad_spottiness = True
    elif args.spottiness_option == "none":
        calc_spot_stats = False

    if args.files_must_exclude is not None:
        ignore_regs = r".*" + re.escape(args.files_must_exclude) + r".*"
    else:
        ignore_regs = None
    if ignore_regs is None or not re.match(ignore_regs, args.filename):
        if args.files_must_include is not None:
            reg = r"(?P<input_directory>.*[\\\/])(?P<name>.*" + re.escape(args.files_must_include) + r".*)(?P<ext>\.tif|\.png)$"
        else:
            reg = r"(?P<input_directory>.*[\\\/])(?P<name>.*)(?P<ext>\.tif|\.png)$"
        results = re.match(reg, args.filename)
        print(
            "Directory: {0}, Name: {1}, Extension: {2}".format(
                results.group("input_directory"),
                results.group("name"),
                results.group("ext"),
            )
        )

        # Set up logging
        logging.getLogger(".".join(__name__.split(".")[:-2])).setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s',datefmt='%m/%d/%Y %H:%M:%S')
        ch.setFormatter(formatter)
        logging.getLogger(".".join(__name__.split(".")[:-2])).addHandler(ch)

        output_directory = add_output_subdirectory(args.output_directory)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        newdirs = ["maps", "masks", "integrals", "stats", "logs"]
        for newdir in newdirs:
            path = os.path.join(output_directory, newdir)
            if not os.path.exists(path):
                os.mkdir(path)

        localname = os.path.splitext(os.path.split(args.filename)[1])[0]
        curtime = time.strftime('%Y_%m_%d_%H_%M_%S')
        logging_filepath = os.path.join(output_directory, 'logs', f'{curtime}_{localname}.log')
        fh = logging.FileHandler(logging_filepath)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logging.getLogger(".".join(__name__.split(".")[:-2])).addHandler(fh)

        # Place all arguments in the log file
        logging.getLogger(__name__).info("Options:"
                                         f"{args.filename=}\n"
                                         f"{args.input_directory=}\n"
                                         f"{args.output_directory=}\n"
                                         f"{args.cache_location=}\n"
                                         f"{args.outlier_option=}\n"
                                         f"{args.min_cluster_area=}\n"
                                         f"{args.min_arc_area=}\n"
                                         f"{args.min_azim_width=}\n"
                                         f"{args.max_q_width=}\n"
                                         f"{args.azim_Q_ratio=}\n"
                                         f"{args.use_radial_grad=}\n"
                                         f"{args.use_azim_grad=}\n"
                                         f"{args.spot_threshold_percentile=}\n"
                                         f"{args.arc_threshold_percentile=}\n"
                                         f"{args.spottiness_option=}\n"
                                         f"{args.calc_azim_Qs=}\n"
                                         f"{args.csim_first_index=}\n"
                                         f"{args.files_must_include=}\n"
                                         f"{args.files_must_exclude=}\n"
                                         )

        timing = []
        timing_names = []
        try:
            run_iteration(
                filename = args.filename,
                input_directory = args.input_directory,
                output_directory = output_directory,
                name = results.group("name"),
                ext = results.group("ext"),
                cache_location = args.cache_location,
                closing_method = "binary_closing",
                calc_outlier = calc_outlier,
                calc_splitting = calc_splitting,
                azim_Q_shape_min = args.azim_Q_ratio,
                calc_spot_stats = calc_spot_stats,
                calc_grad_spottiness = calc_grad_spottiness,
                min_cluster_area = args.min_cluster_area,
                min_arc_area = args.min_arc_area,
                min_azim_width = args.min_azim_width,
                max_q_width = args.max_q_width,
                spot_threshold_percentile = args.spot_threshold_percentile,
                arc_threshold_percentile = args.arc_threshold_percentile,
                calc_azim_Qs = args.calc_azim_Qs,
                use_radial_grad = args.use_radial_grad,
                use_azim_grad = args.use_azim_grad,
                csim_first_index = args.csim_first_index,
                timing = timing,
                timing_names = timing_names,
            )
        except:
            logging.getLogger(__name__).exception(f"Exception in file {args.filename}")
        try:
            logging.getLogger(__name__).info(f"Finished successfully processing {args.filename}.")
            temp_formatter = logging.Formatter('%(message)s')
            fh.setFormatter(temp_formatter)
            ch.setFormatter(temp_formatter)
            for i in range(len(timing[0])):
                logging.getLogger(__name__).info(f"{timing_names[i]}: {timing[0][i]:.2f}s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
        except:
            fh.setFormatter(formatter)
            logging.getLogger(__name__).warning("Problem printing out timing info")
