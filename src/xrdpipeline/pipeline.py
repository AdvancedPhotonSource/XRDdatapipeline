import os
import time

import numpy as np
from scipy import spatial

import skimage as ski

import torch
from PIL import Image

from classification import current_splitting_method, old_splitting_method
from corrections_and_maps import *
from GSASII_imports import *


# recreating xye export function
# TODO: use numpy or the like to write this faster
def Export_xye(name, data, location, error=True):
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
    location += ".chi"
    data_len = len(data[0])
    with open(location,"w") as outfile:
        outfile.write(f"{name} Azm= 0.00\n")
        outfile.write("2-Theta Angle (Degrees)\nIntensity\n")
        outfile.write(f"       {data_len}\n")
        for i in range(data_len):
            outfile.write(f" {data[0][i]:.7e}   {data[1][i]:.7e}\n")


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


def run_iteration(
        filename,
        input_directory,
        output_directory,
        name,
        number,
        cache,
        ext,
        closing_method = "binary_closing",
        return_steps = False,
        calc_outlier = True,
        outChannels = None,
        calc_splitting = True,
        azim_Q_shape_min = 100,
        calc_spottiness = False,
        timing = None,
        timing_names = None,
    ):
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
    if timing is not None:
        timing_0 = time.time()
        local_times = []
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
                name + "-" + number + "_flatfield_correct.tif"
            )
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
    imsave = Image.fromarray(predef_and_nonpositive)
    imsave.save(
        os.path.join(
            output_directory,
            "masks",
            name + "-" + number + "_predef.tif"
        )
    )
    predef_mask_extended = ski.morphology.binary_dilation(
        predef_and_nonpositive, footprint=ski.morphology.square(7)
    )  # extend out by three pixels; use for determining whether something is nearby
    frame_and_predef = np.logical_or(
        predef_and_nonpositive, cache["FrameMask"]
    )
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
                    name + "-" + number + "_closedmask.tif"
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
                    name + "-" + number + "_closedmask.tif"
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
            # return_steps = False
            returned_items = current_splitting_method(
                image_dict["image"].copy(),
                closed_mask,
                cache["pixelQmap"],
                cache["pixelAzmap"],
                cache["gradient"],
                cache["Qbins"],
                return_steps=return_steps,
                interpolate=False,
                azim_Q_shape_min=azim_Q_shape_min,
                calc_spottiness=calc_spottiness,
                predef_mask=nonpositive_mask,
                predef_mask_extended=predef_mask_extended,
                min_arc_area=3,
                timing = local_times,
                timing_names = timing_names,
            )
            # returned_items = old_splitting_method(
            #     image_dict["image"],
            #     closed_mask,
            #     cache["pixelQmap"],
            #     cache["pixelAzmap"],
            #     cache["gradient"],
            #     cache["Qbins"],
            #     return_steps=return_steps,
            #     interpolate=False,
            #     azim_Q_shape_min=azim_Q_shape_min,
            #     calc_spottiness=calc_spottiness,
            #     predef_mask=nonpositive_mask,
            #     predef_mask_extended=predef_mask_extended,
            #     timing = local_times,
            #     timing_names = timing_names,
            # )
            # if return_steps and calc_spottiness:
            #     (
            #         split_spots,
            #         split_arcs,
            #         spots_table,
            #         base_arc,
            #         qgrad_arc,
            #         azim_grad_2,
            #         radial_grad_2,
            #         percents,
            #         num_spots,
            #         num_maxima,
            #         num_spot_maxima
            #     ) = returned_items
            # elif return_steps:
            #     (
            #         split_spots,
            #         split_arcs,
            #         spots_table,
            #         base_arc,
            #         qgrad_arc,
            #         azim_grad_2,
            #         radial_grad_2,
            #     ) = returned_items
            # elif calc_spottiness:
            #     (
            #         split_spots,
            #         split_arcs,
            #         spots_table,
            #         percents,
            #         num_spots,
            #         num_maxima,
            #         num_spot_maxima
            #     ) = returned_items
            # else:
            #     (
            #         split_spots,
            #         split_arcs,
            #         spots_table,
            #     ) = returned_items
            if calc_spottiness:
                (
                    split_spots,
                    split_arcs,
                    spots_table_df,
                    spots_table_grad,
                ) = returned_items
            else:
                (
                    split_spots,
                    split_arcs,
                ) = returned_items
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
            # if return_steps:
            #     imsave = Image.fromarray(base_arc)
            #     imsave.save(
            #         os.path.join(
            #             output_directory,
            #             "masks",
            #             name + "-" + number + "_qwidth_arc.tif"
            #         )
            #     )
            #     imsave = Image.fromarray(qgrad_arc)
            #     imsave.save(
            #         os.path.join(
            #             output_directory,
            #             "masks",
            #             name + "-" + number + "_qgrad_arc.tif"
            #         )
            #     )
            #     imsave = Image.fromarray(azim_grad_2)
            #     imsave.save(
            #         os.path.join(
            #             output_directory,
            #             "grads",
            #             name + "-" + number + "_azim_grad_2.tif"
            #         )
            #     )
            #     imsave = Image.fromarray(radial_grad_2)
            #     imsave.save(
            #         os.path.join(
            #             output_directory,
            #             "grads",
            #             name + "-" + number + "_radial_grad_2.tif"
            #         )
            #     )
            if timing is not None:
                timing_1 = time.time()
                local_times.append(timing_1-timing_0)
                timing_name = "Total mask splitting"
                if timing_name not in timing_names:
                    timing_names.append(timing_name)
                timing_0 = time.time()


    # integrate
    base_mask = frame_and_predef | cache["AzimMask"]
    hist_base = pytorch_integrate(
        image_dict["image"],
        base_mask,
        cache["tth_idx"],
        cache["tth_val"],
        cache["raveled_pol"],
        cache["raveled_dist"],
        cache["tth_size"],
    )
    if calc_outlier:
        hist_closed = pytorch_integrate(
            image_dict["image"],
            np.logical_or(closed_mask, base_mask),
            cache["tth_idx"],
            cache["tth_val"],
            cache["raveled_pol"],
            cache["raveled_dist"],
            cache["tth_size"],
        )
        if calc_splitting:
            hist_closedspotsmasked = pytorch_integrate(
                image_dict["image"],
                np.logical_or(split_spots, base_mask),
                cache["tth_idx"],
                cache["tth_val"],
                cache["raveled_pol"],
                cache["raveled_dist"],
                cache["tth_size"],
            )
            hist_closedarcsmasked = pytorch_integrate(
                image_dict["image"],
                np.logical_or(split_arcs, base_mask),
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
    Export_chi(
        # name + "-" + number + "_base",
        name + "-" + number + ".tif",
        hist_base.T,
        integral_file_base + "_base",
        # error=False,
    )
    if calc_outlier:
        Export_chi(
            # name + "-" + number + "_closed",
            name + "-" + number + ".tif",
            hist_closed.T,
            integral_file_base + "_closed",
            # error=False,
        )
        if calc_splitting:
            Export_chi(
                # name + "-" + number + "_closedspotsmasked",
                name + "-" + number + ".tif",
                hist_closedspotsmasked.T,
                integral_file_base + "_closedspotsmasked",
                # error=False,
            )
            Export_chi(
                # name + "-" + number + "_closedarcsmasked",
                name + "-" + number + ".tif",
                hist_closedarcsmasked.T,
                integral_file_base + "_closedarcsmasked",
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
    if calc_outlier:
        # if calc_splitting:
            # spots stats
            # spots_table.to_csv(stats_prefix + "-" + number + "_spots_stats.csv")
            
            # ~ 950 KB for table
            # 2d histogram: area, Q position
            # ~7800 KB for 1000x1000 bin histogram
            # 81 KB for 100x100 bin histogram

            # cutting this out for now
            # hist, x_edges, y_edges = np.histogram2d(
            #     spots_table["area"].values, spots_table["intensity_mean"].values, 100
            # )
            # with open(
            #     stats_prefix + "-" + number + "_spots_hist.npy", "wb"
            # ) as outfile:
            #     np.save(outfile, hist)
            #     np.save(outfile, x_edges)
            #     np.save(outfile, y_edges)

        # spottiness
        if calc_spottiness:
            # with open(
            #     stats_prefix + "-" + number + "_spottiness.npy", "wb"
            # ) as outfile:
            #     np.save(outfile, percents)
            #     np.save(outfile, num_spots)
            #     np.save(outfile, num_maxima)
            #     np.save(outfile, num_spot_maxima)
            #     np.save(outfile, cache["QbinEdges"])
            spots_table_df.to_csv(stats_prefix + "-" + number + "_spots_stats_df.csv")
            spots_table_grad.to_csv(stats_prefix + "-" + number + "_spots_stats_grad.csv")
            qbins_filename = stats_prefix + "_qbinedges.npy"
            if not os.path.exists(qbins_filename):
                # print(cache["QbinEdges"])
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
    prev_number = ""
    number_int_prev = int(number) - 1
    if number_int_prev < 0:
        # first image (00000) will have no previous image; just compare to self
        prev_number = number
    else:
        # turn int back to '00001' format, padded to 5 digits
        prev_number = f"{number_int_prev:05}"
    try:
        previous_image_name = os.path.join(input_directory, name + "-" + prev_number + ext)
        # print(previous_image_name)
        previous_image = ski.io.imread(
            previous_image_name
        ).astype(np.float32)
    except:
        print("Cannot find previous image for cosine similarity; using current instead.")
        previous_image = image_dict["image"].astype(np.float32)
    if os.path.exists(
        os.path.join(input_directory, name + "-00000" + ext)
    ):
        first_image = ski.io.imread(
            os.path.join(input_directory, name + "-00000" + ext)
        ).astype(np.float32)
    elif os.path.exists(
        os.path.join(input_directory, name + "-00000-00000" + ext)
    ):
        first_image = ski.io.imread(
            os.path.join(input_directory, name + "-00000-00000" + ext)
        ).astype(np.float32)
    elif os.path.exists(
        os.path.join(input_directory, name[:-6] + "-00000" + ext)
    ):
        first_image = ski.io.imread(
            os.path.join(input_directory, name[:-6] + "-00000" + ext)
        ).astype(np.float32)
    else:
        print("Cannot find first image for cosine similarity; using current instead.")
        first_image = image_dict["image"].astype(np.float32)
    csim_f = 1 - spatial.distance.cosine(
        np.array(image_dict["image"], dtype=np.float32).ravel(),
        first_image.ravel(),
    )
    csim_p = 1 - spatial.distance.cosine(
        np.array(image_dict["image"], dtype=np.float32).ravel(),
        previous_image.ravel(),
    )
    with open(stats_prefix + "-" + number + "_csim.txt", "w") as outfile:
        outfile.write(
            "{first:0.9f}\t{prev:0.9f}\n".format(first=csim_f, prev=csim_p)
        )
    # Also not safe for multiprocessing. Need to grab number-1 when it exists
    # cache["Previous image"] = image_dict["image"]

    if timing is not None:
        timing_1 = time.time()
        local_times.append(timing_1-timing_0)
        timing_name = "Cosine Similarity"
        if timing_name not in timing_names:
            timing_names.append(timing_name)
        timing.append(local_times)
    