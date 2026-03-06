"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the regex used for finding specific file names and other similar checks.
"""

import os
import re
from enum import Enum

ImageNumberStyle = Enum('ImageNumberStyle', names=[('NoNumber',0),('Default',1),('NumberOnly',2)])
image_number_style = {
    ImageNumberStyle.NoNumber: r"",
    ImageNumberStyle.Default: r"\d{5}|\d{5}[_\-]\d{5}",
    ImageNumberStyle.NumberOnly: r"\d+"
}

def add_output_subdirectory(directory, subdirectory="XRDdatapipeline_output"):
    """
    Checks whether the specified subdirectory is at the end of the specified directory.
    If not, it is appended to the directory.

    :param directory: [Output] directory to check
    :param subdirectory: Subdirectory to append, if it is not already the last part of the directory string
    """
    if os.path.split(directory)[1] != "":
        if os.path.split(directory)[1] != subdirectory:
            return os.path.join(directory, subdirectory)
        else:
            return directory
    else:
        newpath = os.path.split(directory)[0]
        if os.path.split(newpath)[1] != subdirectory:
            return os.path.join(newpath, subdirectory)
        else:
            return os.path.abspath(newpath)

def check_valid_image(image, regex):
    if re.match(regex, image):
        return True
    else:
        return False

def split_name_number(image_name, number_style = ImageNumberStyle.Default):
    """
    Attempts to split the key part of the name from the appended number for a given image name and number style.
    Returns the substrings of the split name and number.

    :param image_name: Name of the image without its extension or any directories.
    :param number_style: An element of the enum ImageNumberStyle describing the type of appended number to look for.
    """
    num_regex = image_number_style[number_style]
    # Regex explanation:
    # Parentheses provide match groups; if the string matches the whole regex pattern, the substrings of match groups can be pulled out separately
    # (?P<name_of_match_group>things_to_match) The ?P<name_of_match_group> is optional, but helps sort the groups. A ? at the end means it is optional.
    # name: .* Any number of any characters. This is greedy: it will grab as much of the string as possible.
    # [_\-]? 0-1 instances of _ or - (- is escaped)
    # number: regex is provided by the image number style, could be any number of digits (\d+) or sets of five digits, etc.
    # $: end of string
    # Trying to account for both full path name and immediate file name w/o ext by using optional groups will lead to problems in the NoNumber case;
    # the greedy name field will grab the ext.
    reg_image = r"(?P<name>.*)[_\-]?(?P<number>" + num_regex + r")$"
    results = re.match(reg_image, image_name)
    if results is not None:
        name = results.group("name")
        # Cut trailing - or _
        # if name[-1] == "_" or name[-1] == "-":
        #     name = name[:-1]
        number = results.group("number")
        return name, number
    else:
        return image_name, False

def find_name_number(image_name):
    """
    Attempts to split the key part of the name from the appended number for a given image name.
    Cycles through the available number styles given by the enum ImageNumberStyle.
    Returns the substrings of the split name and number followed by the number style used.

    :param image_name: Name of the image without its extension or any directories.
    """
    for style in ImageNumberStyle:
        if style == ImageNumberStyle.NoNumber:
            continue
        name, number = split_name_number(image_name, style)
        if number != False:
            break
    if number == False:
        name, number = split_name_number(image_name, ImageNumberStyle.NoNumber)
        # number will be an empty string
        style = ImageNumberStyle.NoNumber
    return name, number, style

