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

from pipeline.pipeline_iteration import run_iteration
from pipeline.cache_creation import create_cache


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
        azim_Q_ratio,
        outlier_option,
        spottiness_option,
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
    reg_image = r"(?P<input_directory>.*[\\\/])(?P<name>.*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
    if (files_must_include is not None) and (files_must_include.strip() != ""):
        reg_include = r"(?P<input_directory>.*[\\\/])(?P<name>.*" + re.escape(files_must_include) + r".*)[_\-](?P<number>\d{5}|\d{5}[_\-]\d{5})(?P<ext>\.tif|\.png)$"
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
                    results.group("number"),
                    results.group("ext"),
                ]
            )
    if len(queue) == 0:
        print("Found no files in the directory matching the include and exclude requirements.")
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
        filename, name, number, ext = queue.popleft()
        print(filename)
        run_iteration(
            filename=filename,
            input_directory=input_directory,
            output_directory=output_directory,
            name=name,
            number=number,
            ext=ext,
            cache_location=cache_location,
            cache=cache,
            calc_outlier=calc_outlier,
            calc_splitting=calc_splitting,
            azim_Q_shape_min=azim_Q_ratio,
            calc_spot_stats=calc_spot_stats,
            calc_grad_spottiness=calc_grad_spottiness,
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
    # parser.add_argument("--outlier_mad_mult", type=float, default=3.0, help="Multiplier of median absolute deviation to use when considering a value an outlier.")
    parser.add_argument("--outlier_mad_mult", type=float, help="Multiplier of median absolute deviation to use when considering a value an outlier. Default is 3.")
    parser.add_argument("--n_mask_bins", type=int, help="Number of bins used when calculating outliers.")
    # parser.add_argument("-a", "--azim_Q_ratio", type=int, default=100, help="Azimuthal to Q width ratio used for classifying spots.")
    parser.add_argument("-a", "--azim_Q_ratio", type=int, help="Azimuthal to Q width ratio used for classifying spots. Default is 100.")
    parser.add_argument("--outlier_option", choices=["splitting", "outlier_only", "none"], default="splitting", help="Choose whether to perform no outlier masking, outlier masking only, or outlier masking with spot/texture splitting.")
    parser.add_argument("--spottiness_option", choices=["spot_and_gradient","spot_area_only","none"], default="spot_area_only", help="Choose whether to perform spottiness statistics calculations.")
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
        if "XRDdatapipeline_output" not in args.output_directory:
            output_directory = os.path.abspath(args.output_directory)
            output_directory = os.path.join(output_directory, "XRDdatapipeline_output")
        else:
            output_directory = os.path.abspath(args.output_directory)
    else:
        output_directory = None
    if args.imctrl:
        if os.path.exists(os.path.abspath(args.imctrl)):
            imgctrl = os.path.abspath(args.imctrl)
        elif os.path.exists(os.path.join(input_directory, args.imctrl)):
            imgctrl = os.path.join(input_directory, args.imctrl)
        else:
            print(
                "Image control file not found in this directory or in specified directory."
            )
            imgctrl = None
    else:
        imgctrl = None


    if args.no_ui:
        if (args.input_directory is None) or (args.output_directory is None) or (args.imctrl is None):
            print("When launching in no_ui mode, an input directory, output directory, and image control file are required.")
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
                azim_Q_ratio=args.azim_Q_ratio,
                outlier_option=args.outlier_option,
                spottiness_option=args.spottiness_option,
                files_must_include=args.files_must_include,
                files_must_exclude=args.files_must_exclude,
            )
    else:
        import PySide6
        from pyqtgraph.Qt import QtWidgets
        from pipeline.pipeline_UI import main_window

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
            azim_Q_ratio=args.azim_Q_ratio,
            outlier_option=args.outlier_option,
            spottiness_option=args.spottiness_option,
            files_must_include=args.files_must_include,
            files_must_exclude=args.files_must_exclude,
        )
        sys.exit(app.exec())

