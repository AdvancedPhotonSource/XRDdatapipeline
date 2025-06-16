from PIL import Image
from GSASII_imports import *
import torch
import time


def get_Qmap(Tmap, wavelength):
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


def get_azimbands(azmap, numChansAzim):
    dazim = (360) / numChansAzim
    azimband = np.array(azmap / dazim, dtype=np.int32)
    return azimband


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


def run_cache(filename, input_directory, output_directory, imctrlname, blkSize, imgmaskname=None, flatfield=None, esdMul=3.0):
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

    # flatfield_image = np.ones_like(cache["image"])
    flatfield_image = None
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
    if flatfield_image is not None:
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
    _, tifdata, _, _ = GetTifData(filename)
    image_dict["Image Controls"]["pixelSize"] = tifdata["pixelSize"]
    cache["Image Controls"]["pixelSize"] = tifdata["pixelSize"]
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
    cache["esdMul"] = esdMul
    image_dict["Masks"]["SpotMask"]["esdMul"] = esdMul
    numChansAzim = 360
    cache["azimband"] = get_azimbands(cache["pixelAzmap"], numChansAzim)

    # numChans
    LUtth = np.array(cache["Image Controls"]["IOtth"])
    wave = cache["Image Controls"]["wavelength"]
    dsp0 = wave / (2.0 * sind(LUtth[0] / 2.0))
    dsp1 = wave / (2.0 * sind(LUtth[1] / 2.0))
    x0 = GetDetectorXY2(dsp0, 0.0, cache["Image Controls"])[0]
    x1 = GetDetectorXY2(dsp1, 0.0, cache["Image Controls"])[0]
    if not np.any(x0) or not np.any(x1):
        raise Exception
    numChans = int(1000 * (x1 - x0) / cache["Image Controls"]["pixelSize"][0]) // 2
    cache["numChans"] = numChans

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
        cache["Image Controls"]["outChannels"], # could use numchans as calc'd by gsasii
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

