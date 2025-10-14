===========
XRDPipeline
===========

..
        Not currently set up with pypi and others; leaving the template in for later.
        .. image:: https://img.shields.io/pypi/v/xrdpipeline.svg
                :target: https://pypi.python.org/pypi/xrdpipeline

        .. image:: https://img.shields.io/travis/AZjk/xrdpipeline.svg
                :target: https://travis-ci.com/AZjk/xrdpipeline

        .. image:: https://readthedocs.org/projects/xrdpipeline/badge/?version=latest
                :target: https://xrdpipeline.readthedocs.io/en/latest/?version=latest
                :alt: Documentation Status




A package for automated XRD data masking and integration.

..
        * Free software: MIT license
        * Documentation: https://xrdpipeline.readthedocs.io.


Features
--------

* Automated analysis pipeline for incoming or existing XRD data
* Outlier masking and classification into spot or texture sources
* UI to display results live

Installation
------------

This software was designed for Python version 3.10.8. Set up and start a `virtual environment`_ in that version, then run:

.. _`virtual environment`: https://docs.python.org/3/library/venv.html

``git clone https://github.com/AdvancedPhotonSource/XRDdatapipeline.git``

..
        The cookiecutter template applied by Miaoqi should let this be pip-installable, but it needs to be registered with pypi first.
        In the meantime, the URL will need to be updated if/when this repository moves.

To install dependencies, next run:

``python -m pip install -r requirements.txt``

These commands should install this module and all of its prerequisites.

..
        fmask and polymask should work fine for Windows, but need to be recompiled in Linux.
        Need to include extra steps, ie installing and running the compiler
        Add extra linux_requirements.txt to install it, add the makefile for compiling

Instructions: Pipeline
----------------------

To launch the analysis pipeline, ensure your virtual environment is running and run:

``python src/xrdpipeline/pipeline_queue.py``

This will open the pipeline UI.
The input directory is where the analysis pipeline will watch for incoming XRD images and add them to the queue to be processed.

The output directory will hold results such as outlier mask files, integrated powder patterns, and other statistical information.

The configuration file may be either an .imctrl (GSASII) or .poni (PyFAI) calibration file.
If using a .poni file, you must manually enter the integration range, number of integration bins, and polarization to correct for.
If using an .imctrl file, you may optionally overwrite the configuration values.

The flat-field file, experimental mask, and bad pixel masks are optional.
It is recommended to make an experimental mask using the ``mask_widget.py`` file to block out sections such as the beamstop.
The pipeline will automatically mask out dead (zero intensity) pixels, but if there is a known detector mask, place it in the bad pixel mask.

Before starting, choose whether you wish to process images already existing in the input directory by selecting the box in the lower left.

Advanced Settings let you customize certain threshold values or skip parts of the pipeline entirely.
If you wish to change a particular value, make sure to also check the box by its name to indicate it is being overwritten.
There is a Restore Defaults button for the Advanced Settings should you wish to return to the defaults for a later run.

When you are ready to begin processing images, hit the Start button.
Pause will finish processing the current image but cause the program to wait before processing any further images;
it will still gather new images into the queue.
Clear queue will cause the program to continue watching the directory, but clear any current images from the queue.
Stop will clear the queue, complete the current process, and clear the cache of information used for each image.
If you wish to adjust any settings, you must stop the process and start it again.

Instructions: Results UI
------------------------

A user interface displaying the results of the analysis pipeline may be run alongside that pipeline to get a live view of its output.
To run this, type:

``python src/xrdpipeline/pyqtgraph_layout.py``

This will open two windows: the UI itself and a smaller window prompting for the input image directory, output analysis results directory, and image control file.

Once these have been selected, the top left of the UI will show the first image in the set of images it finds, overlaid with a series of masks.
You may toggle the visibility of each mask, the visibility of a 2theta circular ring, and the opacity of each mask in the options below it.
You may also toggle the Live Update option, which will automatically update the image to the most recently processed image.
Tapping left and right on the arrow keys will cycle through all images in a particular dataset. Tapping up and down will cycle between datasets.

The top right will show the integrated data for the current image.
The Base Integral has only the experimental and detector masks applied. All integral lines include these masks.
The outlier masked integral uses the full outlier mask, and the texture and spot masked integral only mask out the texture and spot- tagged masks, respectively.
The Texture Phases and Spot Phases show the difference between the base integral with their respective masked integrals; as such, these show the contributions to intensity from the Texture and Spot tagged outliers, respectively.

The lower left shows a contour plot of all integrated lines in the current dataset. This may also be toggled to update itself as more images are processed.
The view may be swapped between any of the available integral lines and defaults to the Outlier Masked integral.

The lower right has multiple tabbed sections, including a copy of the contour plot (to better align with the above integral lines), various statistics on the spot-tagged clusters, the cosine similarity between images, and the User Data section.
The User Data section allows loading in simulated profiles, such as those output by CrystalMaker, to be shown in other graphs. The two current display options are the integral window and spot statistics window.

Instructions: Mask creation widget
----------------------------------

To launch the widget used to make experimental masks, run:

``python src/xrdpipeline/mask_widget.py``

The program will prompt you for a test image then launch its UI.

A series of buttons allow you to load an image control file and to load or save a mask file in the GSASII .immask format.
Loading an image control file will take a moment; when the maximum 2theta range is filled in with a nonzero value, the file has loaded.
Below that are selectors for an intensity range for the mask and, if you have loaded an image control file, selectors for the 2theta range.

To create a new mask object, select the object from the dropdown menu and hit the "New [Object]" button.
Some objects require an image control file to be loaded to use.

The Polygon, Frame, and Arc objects come with a set of draggable handles to modify the object. You may also modify the text of the object table to adjust the exact values.

For Frames and Polygons, click the New Frame/New Polygon button and then start clicking on the image where you want each new polygon vertex.
When you have enough vertices mapped out, click the Complete Frame/Complete Polygon button.
Polygons will mask the interior of the drawn polygon, and a Frame will mask the exterior section. You may only have one Frame, and will not be allowed to create a second.

For Points, click the New Point button and then select a location on the image. Clicking a new location on the image will update this point's location until you click Complete Point and New Point again.
You may also update the exact pixel location in the text section of the object table.
If you save this mask as an immask file, Points will appear as Spots with radius 1.

For Arcs, select New Arc and click a location near the center of where you wish the arc to be. Five draggable handles will appear.
The center handle can be dragged to move the entire arc segment.
Two handles exist for each of the 2theta and azimuthal range.
If you select Preview Mask and do not see an arc appear in one of the draggable arc objects, ensure you have not swapped the inner and outer handles, then select Clear Preview and Preview Mask again.

X Lines, Y Lines, Spots, and Rings do not currently have a UI implementation, but can still be read in and manually set or adjusted.

X Lines and Y Lines require text in the format [pos], where pos is the integer number of the pixel line being masked.

The Preview Mask button lets you preview the current mask. Save Mask will save a .tif file which can be used in the analysis pipeline.

Tutorials
---------

For more information on usage, please see the tutorials_.

.. _tutorials: https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/docs/tutorials.rst

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
