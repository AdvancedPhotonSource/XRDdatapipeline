"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file handles running different sections of the analysis pipeline.
"""

from collections import deque
import argparse
import os, sys
import glob
import re
import logging

from pipeline.pipeline_iteration import run_iteration
from pipeline.cache_creation import create_cache
from general.file_name_definitions import add_output_subdirectory


def launch_no_ui(
        input_directory,
        output_directory,
        imctrl,
        flatfield,
        imgmask,
        bad_pixels,
        tth_integration_range,
        azim_integration_range,
        n_integration_bins,
        polarization,
        csim_first_index,
        outlier_mad_mult,
        n_mask_bins,
        min_cluster_area,
        min_arc_area,
        min_azim_width,
        max_q_width,
        azim_Q_ratio,
        pixel_size,
        outlier_option,
        use_radial_grad,
        use_azim_grad,
        spot_threshold_percentile,
        arc_threshold_percentile,
        spottiness_option,
        calc_azim_Qs,
        files_must_include,
        files_must_exclude,
    ):
    if outlier_mad_mult is None:
        outlier_mad_mult = 3.0
    if csim_first_index is None:
        csim_first_index = 0

    # run along directory
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    newdirs = ["maps", "masks", "integrals", "stats", "logs"]
    if not ((flatfield is None) or (flatfield == "")):
        newdirs.append("flatfield")
    for newdir in newdirs:
        path = os.path.join(output_directory, newdir)  # store maps with the images
        if not os.path.exists(path):
            os.mkdir(path)

    existing_files = sorted(
                glob.glob(input_directory + "/*.tif"),
                # ctime is not platform-independent, so using mtime
                key = os.path.getmtime
            )
    reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)(?P<ext>\.tif|\.png)$"
    if (files_must_include is not None) and (files_must_include.strip() != ""):
        reg_include = r"(?P<input_directory>.*[\\\/])(?P<name>.*" + re.escape(files_must_include) + r".*)(?P<ext>\.tif|\.png)$"
        regs = reg_include
    else:
        regs = reg_image
    ignore_regs = None
    if (files_must_exclude is not None) and (files_must_exclude.strip() != ""):
        ignore_regs = r".*" + re.escape(files_must_exclude) + r".*"

    queue = deque()
    for filename in existing_files:
        results = re.match(regs, filename)
        if results is not None and ignore_regs is not None:
            if re.match(ignore_regs, filename):
                continue
        elif results is not None:
            queue.append(
                [
                    filename,
                    results.group("name"),
                    results.group("ext"),
                ]
            )
    if len(queue) == 0:
        logging.getLogger(__name__).warning("Found no files in the directory matching the include and exclude requirements.")
        return

    # make cache
    cache_location, cache = create_cache(
        cache={},
        filename=queue[0][0],
        imctrlname=imctrl,
        output_directory=output_directory,
        tth_integration_range=tth_integration_range,
        azim_integration_range=azim_integration_range,
        n_integration_bins=n_integration_bins,
        polarization=polarization,
        imgmaskname=imgmask,
        bad_pixels=bad_pixels,
        flatfield=flatfield,
        esdMul=outlier_mad_mult,
        pixSize=pixel_size,
    )
    # run over files
    calc_outlier = True
    calc_splitting = True
    if outlier_option == "none":
        calc_outlier = False
        calc_splitting = False
    elif outlier_option == "outlier_only":
        calc_splitting = False
    if azim_Q_ratio is None:
        azim_Q_ratio = 100

    calc_spot_stats = True
    calc_grad_spottiness = False
    if spottiness_option == "spot_and_gradient":
        calc_grad_spottiness = True
    elif spottiness_option == "none":
        calc_spot_stats = False
    for it in range(len(queue)):
        filename, name, ext = queue.popleft()
        print(filename)
        run_iteration(
            filename=filename,
            input_directory=input_directory,
            output_directory=output_directory,
            name=name,
            ext=ext,
            cache_location=cache_location,
            cache=cache,
            calc_outlier=calc_outlier,
            calc_splitting=calc_splitting,
            min_cluster_area=min_cluster_area,
            min_arc_area=min_arc_area,
            min_azim_width=min_azim_width,
            max_q_width=max_q_width,
            azim_Q_shape_min=azim_Q_ratio,
            use_radial_grad=use_radial_grad,
            use_azim_grad=use_azim_grad,
            spot_threshold_percentile=spot_threshold_percentile,
            arc_threshold_percentile=arc_threshold_percentile,
            calc_spot_stats=calc_spot_stats,
            calc_grad_spottiness=calc_grad_spottiness,
            calc_azim_Qs=calc_azim_Qs,
            csim_first_index=csim_first_index,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="Location of the input image directory")
    parser.add_argument("-o", "--output_directory", help="Location to place the output files from this pipeline")
    parser.add_argument("-c", "--imctrl", help="Image control file")
    parser.add_argument("-f", "--flatfield", help="Flatfield file")
    parser.add_argument("-m", "--imgmask", help="Experimental mask")
    parser.add_argument("-b", "--bad_pixels", help="Detector known bad pixel mask")
    parser.add_argument("-t", "--tth_integration_range", nargs=2, type=float, help= "2theta integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("-z", "--azim_integration_range", nargs=2, type=float, help="Azimuthal integration range, if overriding or not included in the image control file. Provide minimum and maximum values separated by a space.")
    parser.add_argument("--n_integration_bins", type=float, help="Number of bins to use for integration, if overriding or not included in the config file.")
    parser.add_argument("-p", "--polarization", type=float, help="Polarization of the image")
    parser.add_argument("--csim_first_index", help="Numerical index for the file which should be considered first when calculating cosine similarity.")
    parser.add_argument("--outlier_mad_mult", type=float, default=3.0, help="Multiplier of median absolute deviation to use when considering a value an outlier.")
    parser.add_argument("--n_mask_bins", type=int, help="Number of bins used when calculating outliers.")
    parser.add_argument("--pixel_size", nargs=2, type=int, help="Pixel size in um. Provide x and y values separated by a space.")
    parser.add_argument("--min_cluster_area", type=int, default=3, help="Clusters must be larger than this pixel area to be classified as spot or arc")
    parser.add_argument("--min_arc_area", type=int, default=100, help="Minimum cluster area in pixels to be determined a texture arc")
    parser.add_argument("--min_azim_width", type=float, default=0, help="Minimum azimuthal width in degrees for a cluster to be determined a texture arc")
    parser.add_argument("--max_q_width", type=float, default=0.1, help="Maximum Q width for a cluster to be determined a texture arc")
    parser.add_argument("-a", "--azim_Q_ratio", type=int, default=100, help="Azimuthal to Q width ratio used for classifying spots.")
    parser.add_argument("--outlier_option", choices=["splitting", "outlier_only", "none"], default="splitting", help="Choose whether to perform no outlier masking, outlier masking only, or outlier masking with spot/texture splitting.")
    parser.add_argument("--skip_radial_grad", dest="use_radial_grad", action='store_false', help="Skip use of radial second derivative information to check if clusters are on a powder arc")
    parser.add_argument("--skip_azim_grad", dest="use_azim_grad", action='store_false', help="Skip use of azimuthal second derivative information to cut spots from texture arc candidates")
    parser.add_argument("--spot_threshold_percentile", type=float, default=0.1, help="Percentile of the radial second derivative intensities to use as a threshold to find spots in texture arcs")
    parser.add_argument("--arc_threshold_percentile", type=float, default=10, help="Percentile of the radial second derivative intensities to use as a threshold to determine if a cluster is on a powder ring")
    parser.add_argument("--spottiness_option", choices=["spot_and_gradient","spot_area_only","none"], default="spot_area_only", help="Choose whether to perform spottiness statistics calculations.")
    parser.add_argument("--skip_calc_azim_Qs", dest="calc_azim_Qs", action='store_false', help="Do not calculate and save azimuth / Q spans for all clusters")
    parser.add_argument("--files_must_include", help="Process only files in the directory which include the provided string in their name.")
    parser.add_argument("--files_must_exclude", help="Exclude files in the directory which have the provided string in their name.")
    parser.add_argument("-n", "--no_ui", action="store_true", help="Skip the UI and run over files with the specified options (otherwise options will be pre-filled in the UI). Must include input directory, output directory, and image control file to run.")
    args = parser.parse_args()

    if args.flatfield is not None:
        flatfield = os.path.abspath(args.flatfield)
    else:
        flatfield = None
    if args.imgmask is not None:
        imgmask = os.path.abspath(args.imgmask)
    else:
        imgmask = None
    if args.bad_pixels is not None:
        bad_pixels = os.path.abspath(args.bad_pixels)
    else:
        bad_pixels = None
    if args.input_directory:
        input_directory = os.path.abspath(args.input_directory)
    else:
        input_directory = None
    if args.output_directory:
        output_directory = os.path.abspath(args.output_directory)
        output_directory = add_output_subdirectory(output_directory)
    else:
        output_directory = None
    if args.imctrl:
        if os.path.exists(os.path.abspath(args.imctrl)):
            imgctrl = os.path.abspath(args.imctrl)
        elif os.path.exists(os.path.join(input_directory, args.imctrl)):
            imgctrl = os.path.join(input_directory, args.imctrl)
        else:
            logging.getLogger(__name__).warning(
                "Image control file not found in this directory or in the input directory."
            )
            imgctrl = None
    else:
        imgctrl = None

    if args.no_ui:
        if (args.input_directory is None) or (args.output_directory is None) or (imgctrl is None):
            logging.getLogger(__name__).warning("When launching in no_ui mode, an input directory (-i), output directory (-o), and image control file (-c) are required. See the help info (-h) for more information. " \
            "If you are already inputting all three, try putting file and directory names in quotations. If using Windows, also try dropping any trailing slashes in directory names (so they don't escape the end quote).")
        else:
            launch_no_ui(
                input_directory=input_directory,
                output_directory=output_directory,
                imctrl=imgctrl,
                flatfield=flatfield,
                imgmask=imgmask,
                bad_pixels=bad_pixels,
                tth_integration_range=args.tth_integration_range,
                azim_integration_range=args.azim_integration_range,
                n_integration_bins=args.n_integration_bins,
                polarization=args.polarization,
                csim_first_index=args.csim_first_index,
                outlier_mad_mult=args.outlier_mad_mult,
                n_mask_bins=args.n_mask_bins,
                min_cluster_area=args.min_cluster_area,
                min_arc_area=args.min_arc_area,
                min_azim_width=args.min_azim_width,
                max_q_width=args.max_q_width,
                azim_Q_ratio=args.azim_Q_ratio,
                pixel_size=args.pixel_size,
                outlier_option=args.outlier_option,
                use_radial_grad=args.use_radial_grad,
                use_azim_grad=args.use_azim_grad,
                spot_threshold_percentile=args.spot_threshold_percentile,
                arc_threshold_percentile=args.arc_threshold_percentile,
                spottiness_option=args.spottiness_option,
                calc_azim_Qs=args.calc_azim_Qs,
                files_must_include=args.files_must_include,
                files_must_exclude=args.files_must_exclude,
            )
    else:
        try:
            import PySide6
            from pyqtgraph.Qt import QtWidgets
            from pipeline.pipeline_UI import main_window
        except:
            logging.getLogger(__name__).exception("Exception importing Qt libraries and running the UI. Try running with mode -n/--no_ui.")
        else:
            app = QtWidgets.QApplication([])
            window = main_window(
                input_directory=input_directory,
                output_directory=output_directory,
                imctrl=imgctrl,
                flatfield=flatfield,
                imgmask=imgmask,
                bad_pixels=bad_pixels,
                tth_integration_range=args.tth_integration_range,
                azim_integration_range=args.azim_integration_range,
                n_integration_bins=args.n_integration_bins,
                polarization=args.polarization,
                csim_first_index=args.csim_first_index,
                outlier_mad_mult=args.outlier_mad_mult,
                n_mask_bins=args.n_mask_bins,
                min_cluster_area=args.min_cluster_area,
                min_arc_area=args.min_arc_area,
                min_azim_width=args.min_azim_width,
                max_q_width=args.max_q_width,
                azim_Q_ratio=args.azim_Q_ratio,
                pixel_size=args.pixel_size,
                outlier_option=args.outlier_option,
                use_radial_grad=args.use_radial_grad,
                use_azim_grad=args.use_azim_grad,
                spot_threshold_percentile=args.spot_threshold_percentile,
                arc_threshold_percentile=args.arc_threshold_percentile,
                spottiness_option=args.spottiness_option,
                calc_azim_Qs=args.calc_azim_Qs,
                files_must_include=args.files_must_include,
                files_must_exclude=args.files_must_exclude,
            )
            sys.exit(app.exec())
