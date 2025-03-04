import os, sys
import numpy as np
import PySide6
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import tifffile as tf
from PIL import Image
import re
from GSASII_imports import *

pg.setConfigOptions(imageAxisOrder="row-major")


def pixel_to_mm(distance, pixel_size):
    scale = pixel_size / 1000
    return distance * scale

def mm_to_pixel(distance, pixel_size):
    scale = pixel_size / 1000
    return distance / scale


def choose_file():
    # just tiff for now
    image_file_name = QtWidgets.QFileDialog.getOpenFileName(
        None, "Choose Image", ".", "TIFF files (*.tif)"
    )[0]
    return image_file_name


# Modified from GSASIIFiles to drop oldThreshold, change open/close behavior, return dict
def readMasks(filename):
    masks = {}
    with open(filename, "r") as infile:
        S = infile.readline()
        while S:
            if S[0] == "#":
                S = infile.readline()
                continue
            [key,val] = S.strip().split(":", 1)
            if key in ["Points", "Rings", "Arcs", "Polygons", "Frames", "Thresholds", "Xlines", "Ylines"]:
                masks[key] = eval(val)
            S = infile.readline()
    for key in ["Points", "Rings", "Arcs", "Polygons"]:
        masks[key] = masks.get(key, [])
        masks[key] = [i for i in masks[key] if len(i)]
    return masks

# now need to make the 2theta map from the config file
#create and save TA[x] maps
#from savemaps.py by Wenqian Xu
def getmaps(cache, imctrlname, pathmaps, save=True):		# fast integration using the same imctrl and mask
    #TA = G2img.Make2ThetaAzimuthMap(imctrls,(0,imctrls['size'][0]),(0,imctrls['size'][1]))    #2-theta array, 2880 according to detector pixel numbers
    imctrls = read_imctrl(imctrlname)
    if cache["size"] == (2880,2880):
        cache["pixelSize"] = (150,150)
    imctrls["pixelSize"] = cache["pixelSize"]
    cache["center"] = [0,0]
    cache["center"][0] = imctrls["center"][0]*1000/(imctrls["pixelSize"][0])
    cache["center"][1] = imctrls["center"][1]*1000/(imctrls["pixelSize"][1])
    imctrls["det2theta"] = 0.0
    cache["Image Controls"] = imctrls
    TA = Make2ThetaAzimuthMap(imctrls,(0,cache["size"][0]),(0,cache["size"][1]))
    cache["pixelTAmap"] = TA[0]
    cache["pixelAzmap"] = TA[1]
    cache["pixelsampledistmap"] = TA[2]
    cache["polscalemap"] = TA[3]
    if save:
        # imctrlname = imctrlname.split("\\")[-1].split("/")[-1]
        imctrlname = os.path.split(imctrlname)[1]
        path1 =  os.path.join(pathmaps,imctrlname)
        im = Image.fromarray(TA[0])
        im.save(os.path.splitext(path1)[0] + "_2thetamap.tif")
        im = Image.fromarray(TA[1])
        im.save(os.path.splitext(path1)[0] + "_azmmap.tif")
        im = Image.fromarray(TA[2])
        im.save(os.path.splitext(path1)[0] + "_pixelsampledistmap.tif")
        im = Image.fromarray(TA[3])
        im.save(os.path.splitext(path1)[0] + "_polscalemap.tif")
    return


def read_imctrl(imctrlname):
    image_controls = {}
    with open(imctrlname,'r') as imctrlfile:
        lines = imctrlfile.readlines()
        LoadControls(lines,image_controls)
    return image_controls


def get_save_file_location(ext):
    location = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Save as...",
        ".",
        ext,
    )
    start, typed_end = os.path.splitext(location[0])
    if start == "":
        return None
    if (typed_end != "") and typed_end != ext:
        print("Typed extension does not match required file type.")
    filename = start + ext
    print(f"Saving {filename}")
    return filename


class Point(pg.QtCore.QPoint):
    def __init__(self, image_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.verifyState()
        self.table_label = QtWidgets.QTableWidgetItem("Point")
        self.table_item = QtWidgets.QTableWidgetItem(
            "(" + str(self.x()) + ","
            + str(self.y()) + ")"
        )

    def verifyState(self):
        # print("verifying")
        # print(self.x(),self.y(), self.image_size)
        if self.x() > self.image_size[0]:
            self.setX(self.image_size[0])
        if self.x() < 0:
            self.setX(0)
        if self.y() > self.image_size[1]:
            self.setY(self.image_size[1])
        if self.y() < 0:
            self.setY(0)
    
    def updateFromTable(self):
        matches = re.findall(r"(?P<x>\d+\.*\d*),(?P<y>\d+\.*\d*)",self.table_item.text())
        # print(matches)
        self.setX(int(matches[0][0]))
        self.setY(int(matches[0][1]))
        self.verifyState()
        self.table_item.setText(
            "(" + str(self.x()) + ","
            + str(self.y()) + ")"
        )

    def setMouseHover(self,isHovered):
        # currently dummy, should highlight location
        return

class Polygon(pg.PolyLineROI):
    def __init__(self, image_size, *args, isFrame=False, **kwargs):
        self.has_initialized = False
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        # setPoints() is implemented in PolyLineROI
        # add point with:
        # addFreeHandle(point)
        # addSegment(self.handles[-2]['item'],self.handles[-1]['item'])
        # find points with self.handles
        # self.addFreeHandle(QtCore.QPoint(0,0)) # maxBounds are set for only the first point; resets with clearPoints()
        # self.clearPoints()
        self.has_initialized = True
        self.isFrame = isFrame
        if self.isFrame:
            self.table_label = QtWidgets.QTableWidgetItem("Frame")
        else:
            self.table_label = QtWidgets.QTableWidgetItem("Polygon")
        self.table_item = QtWidgets.QTableWidgetItem("")
        # bottomLeft = QtCore.QPoint(0,0)
        # topRight = QtCore.QPoint(image_size[0],image_size[1])
        # print(bottomLeft,topRight)
        # self.maxBounds = QtCore.QRect(bottomLeft,topRight)
        # link 'isSelected' from table_item to hovering color for ROI
        # saveState() to get tuple versions of coords for saving to disk
        # renderShapeMask() for, well, mask

    # base checkPointMove() is always True
    # moving too quickly off the edge will have the handle not quite reach the edge; need more than this
    # def checkPointMove(self,handle,pos,modifiers):
    #     # keep in bounds of image
    #     point = self.getViewBox().mapSceneToView(pos)
    #     if point.x() < 0 or point.y() < 0 or point.x() > self.image_size[0] or point.y() > self.image_size[1]:
    #     # if pos.x() < 0 or pos.y() < 0 or pos.x() > self.image_size[0] or pos.y() > self.image_size[1]:
    #         return False
    #     else:
    #         return True

    # reimplement setPoints to set the boundaries on the first point
    def setPoints(self, points, closed=None):
        if len(points) > 0:
            # print("point0: ",points[0],type(points[0]))
            point0 = QtCore.QPoint(int(points[0][0]), int(points[0][1]))
            # print("point0 point: ", point0, type(point0))
            bottomLeft = QtCore.QPoint(0, 0) - point0
            # the +1 for topRight is to let it visually move to the edge
            # mask slicing is modified to shift back by one if it's out of bounds
            topRight = (
                QtCore.QPoint(self.image_size[0] + 1, self.image_size[1] + 1) - point0
            )
            # print(bottomLeft,topRight)
            self.maxBounds = QtCore.QRect(
                bottomLeft, topRight
            )  # only affects first point for some reason
            # now get it to affect all points, or rather, decrease the size by the furthest points
            if len(points) > 1:
                xs = list(zip(*points))[0]
                ys = list(zip(*points))[1]
                xmin = np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)
                # print(xmin,xmax,ymin,ymax)
                # the +1 for topRight is to let it visually move to the edge
                # mask slicing is modified to shift back by one if it's out of bounds
                top = int(self.image_size[1] + 1 - ymax)
                bottom = int(0 - ymin)
                left = int(0 - xmin)
                right = int(self.image_size[0] + 1 - xmax)
                bottomLeft = QtCore.QPoint(left, bottom)
                topRight = QtCore.QPoint(right, top)
                print(QtCore.QRect(bottomLeft, topRight))
                self.maxBounds = QtCore.QRect(bottomLeft, topRight)

        super().setPoints(points, closed)
        if self.has_initialized:
            self.table_item.setText(str(points))


    # reimplement movePoint() to go to edge if out of bounds
    def movePoint(self, handle, pos, modifiers=None, finish=True, coords="parent"):
        ## called by Handles when they are moved.
        ## pos is the new position of the handle in scene coords, as requested by the handle.
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        newState = self.stateCopy()
        index = self.indexOfHandle(handle)
        h = self.handles[index]
        p0 = self.mapToParent(h["pos"] * self.state["size"])
        p1 = pg.Point(pos)

        if coords == "parent":
            pass
        elif coords == "scene":
            p1 = self.mapSceneToParent(p1)
        else:
            raise Exception(
                "New point location must be given in either 'parent' or 'scene' coordinates."
            )

        # Add check for p1 out of bounds; move it back
        # # Doesn't visually move the handle back, so use with verifyPoints() to snap them where they should be
        print(p1)
        p1 = p1.toPoint()
        if p1.x() < 0:
            p1.setX(0)
        elif p1.x() > self.image_size[0]:
            p1.setX(self.image_size[0])
        if p1.y() < 0:
            p1.setY(0)
        elif p1.y() > self.image_size[1]:
            p1.setY(self.image_size[1])
        print(p1)

        ## Handles with a 'center' need to know their local position relative to the center point (lp0, lp1)
        if "center" in h:
            c = h["center"]
            cs = c * self.state["size"]
            lp0 = self.mapFromParent(p0) - cs
            lp1 = self.mapFromParent(p1) - cs

        if h["type"] == "t":
            snap = (
                True
                if (modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
                else None
            )
            self.translate(p1 - p0, snap=snap, update=False)

        elif h["type"] == "f":
            newPos = self.mapFromParent(p1)
            print(newPos)
            h["item"].setPos(newPos)
            h["pos"] = newPos
            self.freeHandleMoved = True

        elif h["type"] == "s":
            ## If a handle and its center have the same x or y value, we can't scale across that axis.
            if h["center"][0] == h["pos"][0]:
                lp1[0] = 0
            if h["center"][1] == h["pos"][1]:
                lp1[1] = 0

            ## snap
            if self.scaleSnap or (
                modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                lp1[0] = round(lp1[0] / self.scaleSnapSize) * self.scaleSnapSize
                lp1[1] = round(lp1[1] / self.scaleSnapSize) * self.scaleSnapSize

            ## preserve aspect ratio (this can override snapping)
            if h["lockAspect"] or (modifiers & QtCore.Qt.KeyboardModifier.AltModifier):
                # arv = Point(self.preMoveState['size']) -
                lp1 = lp1.proj(lp0)

            ## determine scale factors and new size of ROI
            hs = h["pos"] - c
            if hs[0] == 0:
                hs[0] = 1
            if hs[1] == 0:
                hs[1] = 1
            newSize = lp1 / hs

            ## Perform some corrections and limit checks
            if newSize[0] == 0:
                newSize[0] = newState["size"][0]
            if newSize[1] == 0:
                newSize[1] = newState["size"][1]
            if not self.invertible:
                if newSize[0] < 0:
                    newSize[0] = newState["size"][0]
                if newSize[1] < 0:
                    newSize[1] = newState["size"][1]
            if self.aspectLocked:
                newSize[0] = newSize[1]

            ## Move ROI so the center point occupies the same scene location after the scale
            s0 = c * self.state["size"]
            s1 = c * newSize
            cc = self.mapToParent(s0 - s1) - self.mapToParent(pg.Point(0, 0))

            ## update state, do more boundary checks
            newState["size"] = newSize
            newState["pos"] = newState["pos"] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return

            self.setPos(newState["pos"], update=False)
            self.setSize(newState["size"], update=False)

        elif h["type"] in ["r", "rf"]:
            if h["type"] == "rf":
                self.freeHandleMoved = True

            if not self.rotatable:
                return
            ## If the handle is directly over its center point, we can't compute an angle.
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return

            ## determine new rotation angle, constrained if necessary
            ang = newState["angle"] - lp0.angle(lp1)
            if ang is None:  ## this should never happen..
                return
            if self.rotateSnap or (
                modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle

            ## create rotation transform
            tr = QtGui.QTransform()
            tr.rotate(ang)

            ## move ROI so that center point remains stationary after rotate
            cc = self.mapToParent(cs) - (tr.map(cs) + self.state["pos"])
            newState["angle"] = ang
            newState["pos"] = newState["pos"] + cc

            ## check boundaries, update
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setPos(newState["pos"], update=False)
            self.setAngle(ang, update=False)

            ## If this is a free-rotate handle, its distance from the center may change.

            if h["type"] == "rf":
                h["item"].setPos(
                    self.mapFromScene(p1)
                )  ## changes ROI coordinates of handle
                h["pos"] = self.mapFromParent(p1)

        elif h["type"] == "sr":
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return

            ang = newState["angle"] - lp0.angle(lp1)
            if ang is None:
                return
            if self.rotateSnap or (
                modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle

            if self.aspectLocked or h["center"][0] != h["pos"][0]:
                newState["size"][0] = (
                    self.state["size"][0] * lp1.length() / lp0.length()
                )
                if self.scaleSnap:  # use CTRL only for angular snap here.
                    newState["size"][0] = (
                        round(newState["size"][0] / self.snapSize) * self.snapSize
                    )

            if self.aspectLocked or h["center"][1] != h["pos"][1]:
                newState["size"][1] = (
                    self.state["size"][1] * lp1.length() / lp0.length()
                )
                if self.scaleSnap:  # use CTRL only for angular snap here.
                    newState["size"][1] = (
                        round(newState["size"][1] / self.snapSize) * self.snapSize
                    )

            if newState["size"][0] == 0:
                newState["size"][0] = 1
            if newState["size"][1] == 0:
                newState["size"][1] = 1

            c1 = c * newState["size"]
            tr = QtGui.QTransform()
            tr.rotate(ang)

            cc = self.mapToParent(cs) - (tr.map(c1) + self.state["pos"])
            newState["angle"] = ang
            newState["pos"] = newState["pos"] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return

            self.setState(newState, update=False)

        # if finish:
        #     print("Verifying State")
        #     self.verifyState()
        # self.stateChanged(finish=finish)

    def updateFromTable(self):
        matches = re.findall(r'(?P<x>\d+\.*\d*),\s*(?P<y>\d+\.*\d*)',self.table_item.text())
        newpoints = []
        for i in range(len(matches)):
            print(matches[i])
            print(matches[i][0],matches[i][1])
            print(int(float(matches[i][0])),int(float(matches[i][1])))
            x = int(float(matches[i][0]))
            y = int(float(matches[i][1]))
            if x < 0:
                x = 0
            elif x > self.image_size[0]:
                x = self.image_size[0]
            if y < 0:
                y = 0
            elif y > self.image_size[1]:
                y = self.image_size[1]
            newpoints.append((x,y))
        self.setPoints(newpoints)
        

    def verifyState(self):
        # something doesn't properly update when dragging out of bounds
        # this properly, visually sets the points to where they are
        points = [(h[1].x(), h[1].y()) for h in self.getLocalHandlePositions()]
        print(points, str(points))
        self.setPoints(points)

class Arc(pg.ROI):
    center = None
    tthmap = None
    azmap = None
    def __init__(self, **args):
        print(self.center)
        self.path = None
        pg.ROI.__init__(self, pos=self.center, size=(1,1), movable=False, rotatable=False, resizable=False, **args)
        self.sigRegionChanged.connect(self._clearPath)
        self.table_label = pg.QtWidgets.QTableWidgetItem("Arc")
        self.table_item = pg.QtWidgets.QTableWidgetItem()
        # self._addHandles()
        self.mask_data = np.zeros(self.tthmap.shape,dtype=bool)
        self.mask_RGBA = np.zeros((self.tthmap.shape[0],self.tthmap.shape[1],4),dtype=np.uint8)
        self.mask_RGBA[:,:,0] = 255
        self.mask_RGBA[:,:,2] = 255
        
    def initialize_handles(self,point):
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])
        angle = self.azmap[point.y()][point.x()]
        # print(angle)
        self.setPos(point)
        self.setSize((1,1))
        self.setAngle(angle)
        self.addTranslateHandle((0.5,0.5),name="Center_point")
        self.addFreeHandle((5.5,0.5),name="Tth_high")
        self.addFreeHandle((-4.5,0.5),name="Tth_low")
        self.addFreeHandle((0.5,50.5),name="Azim_left")
        self.addFreeHandle((0.5,-49.5),name="Azim_right")
        print(self.getLocalHandlePositions())
        print(self.getHandles())
        print(self.getSceneHandlePositions())
        for handle in self.getHandles():
            handle.buildPath()
            handle.update()

    def checkPointMove(self,handle,pos,modifiers):
        if handle.typ == 't':
            pos = self.getViewBox().mapSceneToView(pos)
            print(pos,pos.x(),pos.y())
            angle = self.azmap[int(pos.y())][int(pos.x())]
            print(angle)
            self.setAngle(angle)
        return True

    def verifyState(self):
        return

    def paint(self,p,*args):
        # super().paint(p,*args)
        p.setRenderHint(
            p.RenderHint.Antialiasing,
            self._antialias
        )
        p.setPen(self.currentPen)
        if len(self.handles) > 0:
            for i in range(len(self.handles)):
                name = self.handles[i]['name']
                if name == "Azim_left":
                    azim_left_pos = self.handles[i]['item'].pos()
                elif name == "Center_point":
                    center_pos = self.handles[i]['item'].pos()
                elif name == "Azim_right":
                    azim_right_pos = self.handles[i]['item'].pos()
                elif name == "Tth_high":
                    tth_high_pos = self.handles[i]['item'].pos()
                elif name == "Tth_low":
                    tth_low_pos = self.handles[i]['item'].pos()
            p.drawLine(azim_left_pos,center_pos)
            p.drawLine(center_pos,azim_right_pos)
            p.drawLine(tth_high_pos,center_pos)
            p.drawLine(center_pos,tth_low_pos)
            scene_points = self.getSceneHandlePositions()
            mapped_points = {}
            for i in range(len(scene_points)):
                mapped_points[scene_points[i][0]] = self.getViewBox().mapSceneToView(scene_points[i][1])
            maxtth = self.tthmap[int(mapped_points['Tth_high'].y())][int(mapped_points['Tth_high'].x())]
            mintth = self.tthmap[int(mapped_points['Tth_low'].y())][int(mapped_points['Tth_low'].x())]
            maxazim = self.azmap[int(mapped_points['Azim_left'].y())][int(mapped_points['Azim_left'].x())]
            minazim = self.azmap[int(mapped_points['Azim_right'].y())][int(mapped_points['Azim_right'].x())]
            # print(mintth, maxtth, minazim, maxazim)

            self.mask_data = self.tthmap >= mintth
            self.mask_data &= self.tthmap <= maxtth
            if maxazim > minazim:
                self.mask_data &= self.azmap >= minazim
                self.mask_data &= self.azmap <= maxazim
            else:
                temp = self.azmap <= maxazim
                temp |= self.azmap >= minazim
                self.mask_data &= temp
            
            self.mask_RGBA[:,:,3] = 175*self.mask_data

            self.tth_center = maxtth-.5*(maxtth-mintth)
            self.startazim = minazim
            self.endazim = maxazim
            self.tthwidth = maxtth-mintth

            # self.table_item.setText("[{tth},[{startazim},{endazim}],{tthwidth}]".format(tth=maxtth-.5*(maxtth-mintth),startazim=minazim,endazim=maxazim,tthwidth=maxtth-mintth))
            self.table_item.setText("[{tth},[{startazim},{endazim}],{tthwidth}]".format(tth=self.tth_center,startazim=self.startazim,endazim=self.endazim,tthwidth=self.tthwidth))

    def updateFromTable(self):
        # checking self.isMoving does not help
        # change to update table when handle is let go
        match = re.match(r'\[(?P<tth>\d+\.*\d*),\s*\[(?P<startazim>\d+\.*\d*),\s*(?P<endazim>\d+\.*\d*)\],\s*(?P<tthwidth>\d+\.*\d*)\]',self.table_item.text())
        # print(match.group('tth'))
        tth_center = float(match.group('tth'))
        tthwidth = float(match.group('tthwidth'))
        tthmin = tth_center - 0.5*tthwidth
        tthmax = tth_center + 0.5*tthwidth
        minazim = float(match.group('startazim'))
        maxazim = float(match.group('endazim'))
        self.tth_center = tth_center
        self.startazim = minazim
        self.endazim = maxazim
        self.tthwidth = tthwidth

        if minazim < 0:
            minazim = 360 + minazim
        elif minazim > 360:
            minazim -= 360
        if maxazim < 0:
            maxazim = 360 + maxazim
        elif maxazim > 360:
            maxazim -= 360
        if minazim <= maxazim:
            center_azim = 0.5*(maxazim+minazim)
        else:
            center_azim = 0.5*(maxazim+minazim-360)
        tth_low = self.find_closest_point(tthmin,center_azim)
        tth_high = self.find_closest_point(tthmax,center_azim)
        center_point = self.find_closest_point(tth_center,center_azim)
        azim_right = self.find_closest_point(tth_center,minazim)
        azim_left = self.find_closest_point(tth_center,maxazim)

        # print(tthmin, tthmax, tth_center, minazim, maxazim)
        # print(tth_low, tth_high, center_point, azim_right, azim_left)

        for handle in self.handles:
            if handle['name'] == "Tth_high":
                handle['item'].movePoint(tth_high)
            elif handle['name'] == "Tth_low":
                handle['item'].movePoint(tth_low)
            elif handle['name'] == "Azim_left":
                handle['item'].movePoint(azim_left)
            elif handle['name'] == "Azim_right":
                handle['item'].movePoint(azim_right)
            elif handle['name'] == "Center_point":
                handle['item'].movePoint(center_point)

    def find_closest_point(self,tth,azim):
        # handle cases near azim=0
        z = (self.tthmap - tth)**2 + (self.azmap-azim)**2
        p_yx = np.unravel_index(np.argmin(z),z.shape)
        # print("closest point:",tth,azim,pxy)
        point = pg.QtCore.QPoint(p_yx[1],p_yx[0])
        # translate to scene coords
        point = self.getViewBox().mapViewToScene(point)
        print(point)
        return point
    
    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])

    @pg.QtCore.Slot()
    def _clearPath(self):
        self.path = None

class Line(pg.ROI):
    def __init__(self,image_size,orientation,*args,**kwargs):
        self.image_size = image_size
        self.table_item = pg.QtWidgets.QTableWidgetItem()
        self.orientation = orientation
        if self.orientation == "horizontal":
            self.position = self.image_size[0]/2
            self.table_label = pg.QtWidgets.QTableWidgetItem("X Line")
        else:
            self.position = self.image_size[1]/2
            self.table_label = pg.QtWidgets.QTableWidgetItem("Y Line")
        super().__init__(pos=self.position,movable=False, rotatable=False, resizable=False,*args,**kwargs)

    def setPosition(self,position):
        self.position = position

    def updateFromTable(self):
        position = int(self.table_item.text())
        self.setPosition(position)
    
    def verifyState(self):
        return

    def paint(self,p,*args):
        p.setRenderHint(
            p.RenderHint.Antialiasing,
            self._antialias
        )
        p.setPen(self.currentPen)
        if self.orientation == "horizontal":
            p.drawLine(pg.QtCore.QPoint(0,self.position),pg.QtCore.QPoint(self.image_size[1],self.position))
        else:
            p.drawLine(pg.QtCore.QPoint(self.position,0),pg.QtCore.QPoint(self.position,self.image_size[0]))

class Ring(pg.CircleROI):
    def __init__(self,image_size,*args,**kwargs):
        super().__init__(pos=(0,0),radius=1,movable=False,rotatable=False,resizable=False,*args,**kwargs)
        self.image_size = image_size
        self.table_label = pg.QtWidgets.QTableWidgetItem("Ring")
        self.table_item = pg.QtWidgets.QTableWidgetItem()
    
    def updateFromTable(self):
        match = re.match(r'\[(?P<center_tth>\d+\.*\d*),\s*(?P<tth_width>\d+\.*\d*)\]',self.table_item.text())
        self.center_tth = float(match.group('center_tth'))
        self.tth_width = float(match.group('tth_width'))

    def verifyState(self):
        return

class Spot(pg.CircleROI):
    def __init__(self,image_size,*args,**kwargs):
        super().__init__(pos=(0,0),radius=1,movable=False,rotatable=False,resizable=False,*args,**kwargs)
        self.image_size = image_size
        self.table_label = pg.QtWidgets.QTableWidgetItem("Spot")
        self.table_item = pg.QtWidgets.QTableWidgetItem()

    def _addHandles(self):
        self.addTranslateHandle([0.5,0.5],name="Center_point")
        self.addScaleHandle([0.5*2.**-0.5 + 0.5, 0.5*2.**-0.5 + 0.5],[0.5,0.5],name="Scale")

    def updateFromTable(self):
        match = re.match(r'\[(?P<x>\d+\.*\d*),\s*(?P<y>\d+\.*\d*),\s*(?P<r>\d+\.*\d*)\]',self.table_item.text())
        self.center = [float(match.group('y')),float(match.group('x'))]
        self.radius = float(match.group('r'))

    def paint(self, p, opt, widget):
        super().paint(p,opt,widget)
        # self.center = 
        # self.table_item.setText()

    def verifyState(self):
        return

class NoImctrlWarning(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

    

class MainWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1000,600)
        self.main_image = MainImage()
        # Read/Write buttons
        self.load_immask_button = QtWidgets.QPushButton(
            "Load immask"
        )  # load in GSASII-compatible immask file
        self.load_immask_button.released.connect(self.load_immask)
        self.save_immask_button = QtWidgets.QPushButton("Save immask")
        self.save_immask_button.released.connect(self.save_immask)
        self.preview_mask_button = QtWidgets.QPushButton(
            "Preview Mask"
        )  # hide UI of all ROIs, precalc and show mask
        self.preview_mask_button.released.connect(self.preview_mask)
        self.save_mask_button = QtWidgets.QPushButton(
            "Save Mask"
        )  # require user to preview first?
        self.save_mask_button.released.connect(self.save_mask)
        # imctrl file options
        self.load_imctrl_button = QtWidgets.QPushButton("Load image control file") # load in GSASII-compatible imctrl file or pyfai poni file
        self.load_imctrl_button.released.connect(self.load_imctrls)
        self.cache = None
        self.hasLoadedConfig = False

        # ROI mask options
        self.object_dropdown = QtWidgets.QComboBox()
        self.objects = ["Frame","Polygon","Point","Arc","X Line","Y Line","Spot","Ring"]
        self.object_dropdown.addItems(self.objects)
        self.object_dropdown.setCurrentIndex(1)
        self.object_dropdown.currentIndexChanged.connect(self.object_type_changed)
        self.add_object_button = QtWidgets.QPushButton("New Polygon")
        self.add_object_button.released.connect(self.add_object_button_pressed)
        self.creating_object = False
        self.remove_polygon_button = QtWidgets.QPushButton("Delete Selected Object")
        self.remove_polygon_button.released.connect(self.delete_selected_polygon)
        # ellipse/circle # not supported in GSASII, unless it's new
        # arc - needs azimuth map

        # tth/q thresholds - needs tth/q map. GSASII has this threshold in the config file.
        self.min_tth_threshold_label = QtWidgets.QLabel("Minimum 2Theta:")
        self.min_tth_threshold = QtWidgets.QDoubleSpinBox()
        self.min_tth_threshold_label.setDisabled(True)
        self.min_tth_threshold.setDisabled(True)
        self.max_tth_threshold_label = QtWidgets.QLabel("Maximum 2Theta:")
        self.max_tth_threshold = QtWidgets.QDoubleSpinBox()
        self.max_tth_threshold_label.setEnabled(False)
        self.max_tth_threshold.setEnabled(False)
        # intensity thresholds
        self.min_intensity_threshold_label = QtWidgets.QLabel("Minimum intensity:")
        self.min_intensity_threshold = QtWidgets.QDoubleSpinBox()
        self.min_intensity_threshold.setMinimum(np.min(self.main_image.image_data))
        self.min_intensity_threshold.setMaximum(np.max(self.main_image.image_data))
        self.min_intensity_threshold.setValue(np.min(self.main_image.image_data))
        # self.min_intensity_threshold.valueChanged.connect(self.min_intensity_threshold_changed)
        self.max_intensity_threshold_label = QtWidgets.QLabel("Maximum intensity:")
        self.max_intensity_threshold = QtWidgets.QDoubleSpinBox()
        self.max_intensity_threshold.setMinimum(np.min(self.main_image.image_data))
        self.max_intensity_threshold.setMaximum(np.max(self.main_image.image_data))
        self.max_intensity_threshold.setValue(np.max(self.main_image.image_data))
        # self.max_intensity_threshold.valueChanged.connect(self.max_intensity_threshold_changed)

        # Scroll area in case people add a lot of objects
        self.object_list = QtWidgets.QScrollArea()
        self.object_list_widget = QtWidgets.QWidget()
        self.object_layout = QtWidgets.QGridLayout()

        # Object lists
        # self.objects = [] # keeping in MainImage for the mouseReleaseEvent
        self.polygons_label = QtWidgets.QLabel("Objects")
        self.polygons_table = QtWidgets.QTableWidget()
        self.setBackground('w')
        self.polygons_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.polygons_table.setColumnCount(2)
        self.number_of_objects = 0
        self.polygons_table.setRowCount(self.number_of_objects)
        # self.polygons_table.itemClicked.connect(self.highlightROI)
        # self.polygons_table.selectionModel().currentRowChanged.connect(self.highlightROI)
        self.polygons_table.currentItemChanged.connect(self.highlightROI)
        # self.polygons_table.enter_pressed.connect(self.table_data_changed)
        # self.polygons_table.cellChanged.connect(self.table_data_changed)

        self.update_objects_from_table_button = QtWidgets.QPushButton("Update Objects from Table")
        self.update_objects_from_table_button.released.connect(self.update_objects_from_table)

        # for testing
        # items are created separate from the table
        # poly1 = QtWidgets.QTableWidgetItem("Polygon 1")
        # poly2 = QtWidgets.QTableWidgetItem("Polygon 2")
        # row/column count must be set to have space before adding
        # self.polygons_table.setColumnCount(1)
        # self.polygons_table.setRowCount(1)
        # Item can be added as long as there's space there
        # self.polygons_table.setItem(0,0,poly1)
        # RowCount can be extended for each item, as long as the table has space before the item is added
        # self.polygons_table.setRowCount(2)
        # self.polygons_table.setItem(1,0,poly2)

        self.object_layout.addWidget(self.min_intensity_threshold_label, 0, 0)
        self.object_layout.addWidget(self.min_intensity_threshold, 0, 1)
        self.object_layout.addWidget(self.max_intensity_threshold_label, 1, 0)
        self.object_layout.addWidget(self.max_intensity_threshold, 1, 1)
        self.object_layout.addWidget(self.min_tth_threshold_label, 2, 0)
        self.object_layout.addWidget(self.min_tth_threshold, 2, 1)
        self.object_layout.addWidget(self.max_tth_threshold_label, 3, 0)
        self.object_layout.addWidget(self.max_tth_threshold, 3, 1)
        self.object_layout.addWidget(self.polygons_label, 4, 0)
        self.object_layout.addWidget(self.object_dropdown, 5, 0)
        self.object_layout.addWidget(self.add_object_button, 5, 1)
        self.object_layout.addWidget(self.polygons_table, 6, 0, 2, 2)
        self.object_layout.addWidget(self.remove_polygon_button, 8, 0)
        self.object_layout.addWidget(self.update_objects_from_table_button, 8, 1)
        self.object_list_widget.setLayout(self.object_layout)
        self.object_list.setWidget(self.object_list_widget)

        self.min_tth_threshold_label.setDisabled(True)
        self.min_tth_threshold.setDisabled(True)

        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.addWidget(self.main_image, 0, 0, 5, 5)
        self.gridlayout.addWidget(self.load_immask_button, 0, 5)
        self.gridlayout.addWidget(self.save_immask_button, 0, 6)
        self.gridlayout.addWidget(self.load_imctrl_button, 0, 7)
        self.gridlayout.addWidget(self.object_list, 1, 5, 4, 3)
        self.gridlayout.addWidget(self.preview_mask_button, 5, 5)
        self.gridlayout.addWidget(self.save_mask_button, 5, 6)
        self.setLayout(self.gridlayout)
        self.show()

    def load_imctrls(self):
        imctrl_file_name = QtWidgets.QFileDialog.getOpenFileName(None,"Choose Image Control File",".","imctrl files (*.imctrl)")[0]
        self.cache = {}
        self.cache['size'] = self.main_image.image_data.shape
        getmaps(self.cache, imctrl_file_name,".", save=False) # use os path
        Arc.tthmap = self.cache['pixelTAmap']
        Arc.azmap = self.cache['pixelAzmap']
        Arc.center = self.cache['center']
        self.min_tth_threshold_label.setEnabled(True)
        self.min_tth_threshold.setEnabled(True)
        self.min_tth_threshold.setMinimum(0)
        self.min_tth_threshold.setMaximum(np.max(self.cache['pixelTAmap']))
        self.min_tth_threshold.setValue(0)
        self.max_tth_threshold_label.setEnabled(True)
        self.max_tth_threshold.setEnabled(True)
        self.max_tth_threshold.setMinimum(0)
        self.max_tth_threshold.setMaximum(np.max(self.cache['pixelTAmap']))
        self.max_tth_threshold.setValue(np.max(self.cache['pixelTAmap']))
        self.hasLoadedConfig = True
        self.imagescale_is_square = False
        if self.cache['pixelSize'][0] == self.cache['pixelSize'][1]:
            self.imagescale_is_square = True

    def object_type_changed(self,evt):
        # ensure previous object loses focus
        self.done_creating()
        # define an enum or the like so these automatically update with new options
        if evt == 0:
            frame_list = self.polygons_table.findItems("Frame",QtCore.Qt.MatchFlag.MatchExactly)
            print(frame_list, len(frame_list))
            if len(frame_list) > 0:
                self.add_object_button.setText("Max 1 Frame")
            else:
                self.add_object_button.setText("New Frame")
        elif evt == 1:
            self.add_object_button.setText("New Polygon")
        elif evt == 2:
            self.add_object_button.setText("New Point")
        elif evt == 3:
            self.add_object_button.setText("New Arc")   
        elif evt == 4:
            self.add_object_button.setText("New X Line")
        elif evt == 5:
            self.add_object_button.setText("New Y Line")
        elif evt == 6:
            self.add_object_button.setText("New Spot")
        elif evt == 7:
            self.add_object_button.setText("New Ring")

    def add_object_button_pressed(self):
        cur_text = self.add_object_button.text()
        if cur_text == "New Polygon":
            self.add_polygon()
            self.add_object_button.setText("Complete Polygon")
            self.creating_object = True
        elif cur_text == "Complete Polygon":
            self.done_creating()
            self.add_object_button.setText("New Polygon")
            self.creating_object = False
        elif cur_text == "New Point":
            self.add_point()
            self.add_object_button.setText("Complete Point")
            self.creating_object = True
        elif cur_text == "Complete Point":
            self.done_creating()
            self.add_object_button.setText("New Point")
            self.creating_object = False
        elif cur_text == "New Arc":
            if not self.hasLoadedConfig:
                QtWidgets.QMessageBox.question(self,"Exit","Please load an image control file before using arcs and rings.",QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Ok)
                return
            self.add_arc()
            self.add_object_button.setText("Complete Arc")
            self.creating_object = True
        elif cur_text == "Complete Arc":
            self.done_creating()
            self.add_object_button.setText("New Arc")
            self.creating_object = False
        elif cur_text == "New Frame":
            self.add_polygon(isFrame=True)
            self.add_object_button.setText("Complete Frame")
            self.creating_object = True
        elif cur_text == "Complete Frame":
            self.done_creating()
            self.add_object_button.setText("Max 1 Frame")
            self.creating_object = False
        elif cur_text == "New X Line":
            self.add_line(orientation="horizontal")
        elif cur_text == "New Y Line":
            self.add_line(orientation="vertical")
        elif cur_text == "New Spot":
            self.add_spot()
        elif cur_text == "New Ring":
            if not self.hasLoadedConfig:
                QtWidgets.QMessageBox.question(self,"Exit","Please load an image control file before using arcs and rings.",QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Ok)
                return
            self.add_ring()

    def add_polygon(self,isFrame = False):
        poly = self.main_image.add_polygon(isFrame = isFrame)
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects - 1, 0, poly.table_label)
        self.polygons_table.setItem(self.number_of_objects - 1, 1, poly.table_item)
        self.main_image.objects.append(poly)
        self.main_image.current_polygon = self.main_image.objects[-1]

    def add_line(self,orientation):
        line = Line(self.main_image.image_data.shape,orientation=orientation)
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects-1,0,line.table_label)
        self.polygons_table.setItem(self.number_of_objects-1,1,line.table_item)
        self.main_image.objects.append(line)

    def add_spot(self):
        spot = Spot(self.main_image.image_data.shape)
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects-1,0,spot.table_label)
        self.polygons_table.setItem(self.number_of_objects-1,1,spot.table_item)
        self.main_image.objects.append(spot)

    def add_ring(self):
        ring = Ring(self.main_image.image_data.shape)
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects-1,0,ring.table_label)
        self.polygons_table.setItem(self.number_of_objects-1,1,ring.table_item)
        self.main_image.objects.append(ring)

    def done_creating(self):
        self.creating_object = False
        self.main_image.current_polygon = None
        self.main_image.current_point = None
        self.main_image.current_arc = None

    def add_point(self):
        point = Point(image_size=self.main_image.image_data.shape)
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects-1,0,point.table_label)
        self.polygons_table.setItem(self.number_of_objects-1,1,point.table_item)
        self.main_image.objects.append(point)
        self.main_image.current_point = self.main_image.objects[-1]

    def add_arc(self):
        arc = self.main_image.add_arc()
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects-1,0,arc.table_label)
        self.polygons_table.setItem(self.number_of_objects-1,1,arc.table_item)
        self.main_image.objects.append(arc)
        self.main_image.current_arc = self.main_image.objects[-1]

    def preview_mask(self):
        if self.preview_mask_button.text() == "Preview Mask":
            self.predef_mask = np.zeros_like(self.main_image.image_data, dtype=bool)
            for i in self.main_image.objects:
                # poly_mask = i.getArrayRegion(np.ones_like(self.main_image.image_data,dtype=bool),self.main_image.image, returnMappedCoords=True)
                # print(poly_mask, poly_mask[0].shape)
                # print(poly_mask.pos()) # should be lower-left corner
                # poly_mask = i.renderShapeMask(2880,2880) # mask is drawn to size given
                # Play with transpose because this doesn't recognize row-major images, but still knows positions by x,y
                if type(i) == Polygon:
                    # print(i.getLocalHandlePositions())
                    # print([tuple(h.pos()) for h in i.getHandles()])
                    # print([(h[1].x(),h[1].y()) for h in i.getLocalHandlePositions()])
                    points = [(h[1].x(), h[1].y()) for h in i.getLocalHandlePositions()]
                    # i.clearPoints()
                    corrected_points = points
                    for it in range(len(corrected_points)):
                        # print(corrected_points[it])
                        if corrected_points[it][0] >= self.main_image.image_data.shape[0]:
                            corrected_points[it] = (
                                self.main_image.image_data.shape[0] - 1,
                                corrected_points[it][1]
                            )
                        if corrected_points[it][1] >= self.main_image.image_data.shape[1]:
                            corrected_points[it] = (
                                corrected_points[it][0],
                                self.main_image.image_data.shape[1] - 1
                            )
                        # print(corrected_points[it])
                    # print("Corrected points:", corrected_points)
                    i.setPoints(corrected_points)
                    # i.stateChanged()
                    array_slice = i.getArraySlice(
                        self.main_image.image_data, self.main_image.image, returnSlice=False
                    )
                    # print(array_slice)
                    # print(np.abs(array_slice[0][1][0]-array_slice[0][1][1]),np.abs(array_slice[0][0][0]-array_slice[0][0][1]))
                    small_poly_mask = i.renderShapeMask(
                        np.abs(array_slice[0][1][0]-array_slice[0][1][1]),
                        np.abs(array_slice[0][0][0]-array_slice[0][0][1])
                    )
                    # small_poly_mask = i.renderShapeMask(np.abs(array_slice[0][0][0]-array_slice[0][0][1]),np.abs(array_slice[0][1][0]-array_slice[0][1][1]))
                    poly_mask = np.zeros_like(self.main_image.image_data,dtype=bool)
                    array_slice = i.getArraySlice(
                        self.main_image.image_data, self.main_image.image
                    )
                    # print(array_slice)
                    # print(array_slice[0][1],array_slice[0][0])
                    # poly_mask[array_slice[0]] = small_poly_mask
                    poly_mask[(array_slice[0][1], array_slice[0][0])] = small_poly_mask
                    if i.isFrame:
                        self.predef_mask = np.logical_or(self.predef_mask, ~poly_mask.T)
                    else:
                        self.predef_mask = np.logical_or(self.predef_mask, poly_mask.T)
                    i.setPoints(points)
                    # print(i.boundingRect(),i.parentBounds(),i.pos())
                elif type(i) == Point:
                    self.predef_mask[i.y(), i.x()] = True
                elif type(i) == Arc:
                    self.predef_mask = np.logical_or(self.predef_mask, i.mask_data)
                elif type(i) == Line:
                    if i.orientation == "horizontal":
                        self.predef_mask[i.position, :] = True
                    else:
                        self.predef_mask[:, i.position] = True
                elif type(i) == Spot:
                    x, y = np.mgrid[
                        0:self.main_image.image_data.shape[0],
                        0:self.main_image.image_data.shape[1]
                    ]
                    dist = np.sqrt(
                        (x-i.center[0])**2
                        + (y-i.center[1])**2
                    )
                    self.predef_mask |= (dist < i.radius)
                elif type(i) == Ring:
                    temp = self.cache["pixelTAmap"] > (i.center_tth - 0.5 * i.tth_width)
                    temp = np.logical_and(temp,self.cache["pixelTAmap"] < (i.center_tth + 0.5 * i.tth_width))
                    self.predef_mask |= temp
                    ~temp

            # intensity thresholds
            below_mins = np.nonzero(
                self.main_image.image_data < self.min_intensity_threshold.value()
            )
            above_maxs = np.nonzero(
                self.main_image.image_data > self.max_intensity_threshold.value()
            )
            # self.predef_mask |= below_mins
            self.predef_mask[below_mins] = True
            # self.predef_mask |= above_maxs
            self.predef_mask[above_maxs] = True

            # tth thresholds
            if self.cache is not None:
                below_mins = np.nonzero(self.cache["pixelTAmap"] < self.min_tth_threshold.value())
                above_maxs = np.nonzero(self.cache["pixelTAmap"] > self.max_tth_threshold.value())
                self.predef_mask[below_mins] = True
                self.predef_mask[above_maxs] = True
            
            # set mask image data
            # self.main_image.predef_mask.setData(np.array(self.predef_mask,dtype=np.uint8))
            self.poly_mask_rgba = (
                np.ones(
                    (self.predef_mask.shape[0], self.predef_mask.shape[1], 4),
                    dtype=np.uint8,
                )
                * 255
            )
            self.poly_mask_rgba[:, :, 3] = self.predef_mask * 255
            self.main_image.predef_mask.updateImage(self.poly_mask_rgba)
            # self.main_image.image.updateImage(np.array(self.predef_mask,dtype=np.uint8))

            self.preview_mask_button.setText("Clear Preview")
        elif self.preview_mask_button.text() == "Clear Preview":
            self.poly_mask_rgba[:, :, 3] = 0
            self.main_image.predef_mask.updateImage(self.poly_mask_rgba)
            self.preview_mask_button.setText("Preview Mask")

    def save_mask(self):
        location = get_save_file_location(".tif")
        if location is not None:
            tf.imwrite(location, self.predef_mask)

    def save_immask(self):
        outfilename = get_save_file_location(".immask")
        if outfilename is not None:
            # GSASII outputs/expects an explicit return to the first point
            polygon_points_list = []
            frame_points_list = []
            arcs_list = []
            xlines = []
            ylines = []
            points = []
            rings_list = []
            for i in self.main_image.objects:
                if type(i) == Polygon:
                    poly_points = i.saveState()["points"]
                    # print(poly_points)
                    poly_points = [
                        [
                            pixel_to_mm(x, self.cache["pixelSize"][0]),
                            pixel_to_mm(y,self.cache["pixelSize"][1])
                        ] for x,y in poly_points
                    ]
                    # print(poly_points)
                    poly_points.append(poly_points[0])
                    if i.isFrame:
                        frame_points_list = poly_points
                    else:
                        polygon_points_list.append(poly_points)
                elif type(i) == Arc:
                    # arcs_list.append(i.table_item.text())
                    # self.table_item.setText("[{tth},[{startazim},{endazim}],{tthwidth}]".format(tth=self.tth_center,startazim=self.startazim,endazim=self.endazim,tthwidth=self.tthwidth))
                    arcs_list.append([
                        i.tth_center,
                        [i.startazim,i.endazim],
                        i.tthwidth
                    ])
                elif type(i) == Line:
                    if i.orientation == "horizontal":
                        xlines.append(int(i.table_item.text()))
                    else:
                        ylines.append(int(i.table_item.text()))
                elif type(i) == Spot:
                    center0_mm = pixel_to_mm(
                        i.center[1],
                        self.cache["pixelSize"][0]
                    )
                    center1_mm = pixel_to_mm(
                        i.center[0],
                        self.cache["pixelSize"][1]
                    )
                    if self.imagescale_is_square:
                        diameter_mm = pixel_to_mm(
                            i.radius * 2,
                            self.cache["pixelSize"][0]
                        )
                    else:
                        print("Warning: Radius may not align with GSASII as image is not square.")
                        diameter_mm = pixel_to_mm(
                            i.radius * 2,
                            np.sqrt(
                                self.cache["pixelSize"][0] ** 2
                                + self.cache["pixelSize"][1] ** 2
                            )
                        )
                    points.append([center0_mm, center1_mm, diameter_mm])
                elif type(i) == Point:
                    x_mm = pixel_to_mm(
                        i.x(),
                        self.cache["pixelSize"][0]
                    )
                    y_mm = pixel_to_mm(
                        i.y(),
                        self.cache["pixelSize"][1]
                    )
                    if self.imagescale_is_square:
                        d_mm = pixel_to_mm(
                            1,
                            self.cache["pixelSize"][0]
                        )
                    else:
                        d_mm = pixel_to_mm(
                            1,
                            np.sqrt(
                                self.cache["pixelSize"][0] ** 2
                                + self.cache["pixelSize"][1] ** 2
                            )
                        )
                    points.append([x_mm, y_mm, d_mm])
                elif type(i) == Ring:
                    rings_list.append([i.center_tth, i.tth_width])

            # outfilename = os.path.join(".", "mask.immask")
            with open(outfilename,'w') as outfile:
                outfile.write("Points:{points}\n".format(points=points))
                outfile.write("Rings:{rings}\n".format(rings=rings_list))
                outfile.write("Arcs:{arcs}\n".format(arcs=arcs_list))
                # print(points_list)
                outfile.write(
                    "Polygons:{polys}\n".format(polys=polygon_points_list)
                    .replace('(','[').replace(')',']')
                )
                outfile.write("Xlines:{xlines}\n".format(xlines=xlines))
                outfile.write("Ylines:{ylines}\n".format(ylines=ylines))
                outfile.write(
                    "Frames:{frames}\n".format(frames=frame_points_list)
                    .replace('(','[').replace(')',']')
                )
                # outfile.write("Thresholds:[({image_min}, {image_max}), [{image_min}, {image_max}]]".format(image_min=np.min(self.main_image.image_data),image_max=np.max(self.main_image.image_data)))
                outfile.write(
                    "Thresholds:[({image_min}, {image_max}), [{image_min}, {image_max}]]".format(
                        image_min=self.min_intensity_threshold.value(),
                        image_max=self.max_intensity_threshold.value(),
                    )
                )

    def load_immask(self):
        print("loading")
        infilename = QtWidgets.QFileDialog.getOpenFileName(
            None, "Choose Image Mask", ".", "Immask files (*.immask)"
        )[0]
        masks = readMasks(infilename)
        print(masks)
        # TODO: prompt for clearing the current table vs appending
        if not self.hasLoadedConfig:
            print("No config loaded. Please load a config.")
            return
        for poly in masks["Polygons"]:
            self.add_polygon()
            print(poly)
            for point in poly:
                p0_pix = int(mm_to_pixel(point[0], self.cache["pixelSize"][0]))
                p1_pix = int(mm_to_pixel(point[1], self.cache["pixelSize"][1]))
                self.main_image.add_polygon_point(QtCore.QPoint(p0_pix, p1_pix))
            self.done_creating()
        if len(masks["Frames"]) > 0:
            self.add_polygon(isFrame=True)
            for point in masks["Frames"]:
                p0_pix = int(mm_to_pixel(point[0], self.cache["pixelSize"][0]))
                p1_pix = int(mm_to_pixel(point[1], self.cache["pixelSize"][1]))
                self.main_image.add_polygon_point(QtCore.QPoint(p0_pix, p1_pix))
            self.done_creating()
        if (len(masks["Points"]) > 0) and not self.imagescale_is_square:
            print(
                "Warning: Radius of spots may not align with GSASII due to the image not being square. "
                "The radius is calculated in pixels here, but in mm in GSASII. "
                "Direct translations between the two would make ellipses."
            )
        for spot in masks["Points"]:
            self.add_spot()
            x, y, d = spot
            x_pix = mm_to_pixel(x, self.cache["pixelSize"][0])
            y_pix = mm_to_pixel(y, self.cache["pixelSize"][1])
            if self.imagescale_is_square:
                r_pix = mm_to_pixel(d/2, self.cache["pixelSize"][0])
            else:
                r_pix = mm_to_pixel(
                    d/2,
                    np.sqrt(
                        self.cache["pixelSize"][0] ** 2
                        + self.cache["pixelSize"][1] **2
                    )
                )
            spot = [x_pix, y_pix, r_pix]
            self.main_image.objects[-1].table_item.setText(str(spot))
            self.main_image.objects[-1].updateFromTable()
        for ring in masks["Rings"]:
            self.add_ring()
            self.main_image.objects[-1].table_item.setText(str(ring))
            self.main_image.objects[-1].updateFromTable()
        for arc in masks["Arcs"]:
            self.add_arc()
            self.main_image.set_arc_point(pg.QtCore.QPoint(0, 0))
            self.done_creating()
            self.main_image.objects[-1].table_item.setText(str(arc))
            self.main_image.objects[-1].updateFromTable()
        for xline in masks["Xlines"]:
            self.add_line(orientation="horizontal")
            self.main_image.objects[-1].table_item.setText(str(xline))
            self.main_image.objects[-1].updateFromTable()
        for yline in masks["Ylines"]:
            self.add_line(orientation="vertical")
            self.main_image.objects[-1].table_item.setText(str(yline))
            self.main_image.objects[-1].updateFromTable()
        # Run 'done creating' one more time just in case
        self.done_creating()
        self.min_intensity_threshold.setValue(masks["Thresholds"][1][0])
        self.max_intensity_threshold.setValue(masks["Thresholds"][1][1])


    def image_changed(self, data):
        # set image data
        self.main_image.image_data = data
        # reset min/max intensity
        self.min_intensity_threshold.setMinimum(np.min(self.main_image.image_data))
        self.min_intensity_threshold.setMaximum(np.max(self.main_image.image_data))
        self.max_intensity_threshold.setMinimum(np.min(self.main_image.image_data))
        self.max_intensity_threshold.setMaximum(np.max(self.main_image.image_data))

    def clear_polygon(self, index):
        # clear points
        self.main_image.objects[index].clearPoints()
        # reduce size of objects list
        self.number_of_objects -= 1
        # remove from table
        self.polygons_table.removeRow(index)
        # remove from objects list 
        del self.main_image.objects[index]

    def clear_arc(self, index):
        self.main_image.objects[index].clearPoints()
        self.number_of_objects -= 1
        self.polygons_table.removeRow(index)
        del self.main_image.objects[index]

    def clear_point(self, index):
        # reduce size of objects list
        self.number_of_objects -= 1
        # remove from table
        self.polygons_table.removeRow(index)
        # remove from objects list
        del self.main_image.objects[index]

    def clear_line(self, index):
        self.number_of_objects -= 1
        self.polygons_table.removeRow(index)
        del self.main_image.objects[index]

    def clear_spot(self, index):
        self.number_of_objects -= 1
        self.polygons_table.removeRow(index)
        del self.main_image.objects[index]

    def clear_ring(self, index):
        self.number_of_objects -= 1
        self.polygons_table.removeRow(index)
        del self.main_image.objects[index]

    def delete_selected_polygon(self):
        # list of singular item, as selection mode is single selection
        selected = self.polygons_table.selectedItems()[0]
        selected_row = selected.row()
        label_text = self.polygons_table.item(selected_row, 0).text()
        # print("Deleting {0} object at row {1}".format(label_text,selected_row))
        if label_text == "Polygon":
            self.clear_polygon(index=selected_row)
        elif label_text == "Frame":
            self.clear_polygon(index=selected_row)
            if self.add_object_button.text() == "Max 1 Frame":
                self.add_object_button.setText("New Frame")
        elif label_text == "Point":
            self.clear_point(index=selected_row)
        elif label_text == "Arc":
            self.clear_arc(index=selected_row)
        elif (label_text == "X Line") or (label_text == "Y Line"):
            self.clear_line(index=selected_row)
        elif label_text == "Spot":
            self.clear_spot(index=selected_row)
        elif label_text == "Ring":
            self.clear_ring(index=selected_row)

    def update_objects_from_table(self):
       for index in range(len(self.main_image.objects)):
            item = self.main_image.objects[index]
            if not self.creating_object:
                item.updateFromTable()

    # def table_data_changed(self,index):
    #     try:
    #         item = self.main_image.objects[index]
    #         if not self.creating_object:
    #             item.updateFromTable()
    #     except IndexError:
    #         return
        
    def highlightROI(self,current,previous):
        # print("Selected {0} object at {1}: {2},{3}".format(self.polygons_table.item(current.row(),0).text(),current.row(),self.polygons_table.item(current.row(),1).text(),current))
        for i in range(self.number_of_objects):
            if (current) and (i == current.row()):
                self.main_image.objects[i].setMouseHover(True)
            else:
                self.main_image.objects[i].setMouseHover(False)


class MainImage(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.view = self.addPlot()
        self.view.setAspectLocked(True)
        self.cmap = pg.colormap.get("gist_earth", source="matplotlib", skipCache=True)
        # self.image_data = np.zeros((2880,2880))
        self.image_data = tf.imread(choose_file())
        self.image = pg.ImageItem(self.image_data)
        self.predef_mask_data = np.zeros(
            (self.image_data.shape[0], self.image_data.shape[1], 4), dtype=np.uint8
        )
        self.predef_mask = pg.ImageItem(self.predef_mask_data, levels=None)
        # self.qpicture = None
        self.current_polygon = None
        self.current_point = None
        self.current_arc = None
        self.objects = []

        self.view.addItem(self.image)
        # self.view.setRange(xRange=(0,self.image_data.shape[0]),yRange=(0,self.image_data.shape[1]))
        # self.view.vb.setLimits(xMin=0,xMax=self.image_data.shape[0],yMin=0,yMax=self.image_data.shape[1])
        # print(self.view.viewRange())
        self.view.addItem(self.predef_mask)
        self.intensityBar = pg.HistogramLUTItem()
        self.intensityBar.setImageItem(self.image)
        self.intensityBar.gradient.setColorMap(self.cmap)
        self.intensityBar.gradient.showTicks(show=False)
        self.addItem(self.intensityBar)

        self.view.scene().sigMouseClicked.connect(self.mouse_click)
        # self.view.scene().sigMouseReleased.connect(self.mouse_release)

        # self.polygons = []
        # self.polygon_points = []
        # self.polygon = Polygon(self.polygon_points, closed=True)
        # self.polygon = pg.PolyLineROI(self.polygon_points,closed=True)
        # self.view.addItem(self.polygon)

        # self.polygon = QtGui.QPainter.drawPolygon()
        # self.show()

    def add_polygon(self, isFrame = False):
        # bounds = QtCore.QRect(0,2880,2880,2880)
        # bounds = QtCore.QRect(QtCore.QPoint(2880,0),QtCore.QPoint(0,2880))
        bounds = QtCore.QRect()
        bounds.setLeft(0)
        bounds.setRight(self.image_data.shape[0])
        bounds.setBottom(0)
        bounds.setTop(self.image_data.shape[1])
        # print(bounds, bounds.bottomLeft(),bounds.bottomRight(),bounds.topLeft(),bounds.topRight())
        # y goes from 0 to 2880, but x goes from 1 to 2879?
        # print(self.image_data.shape) # 2880,2880
        # print(self.image.boundingRect())
        # need to transform bounding rect back to proper coords
        # polygon = Polygon(positions=[],closed=True,translateSnap=True,rotatable=False,maxBounds=self.image.boundingRect())
        polygon = Polygon(
            positions=[],
            closed=True,
            translateSnap=True,
            rotatable=False,
            image_size=self.image_data.shape,
            isFrame=isFrame,
            parent=self.view
        )
        
        self.view.addItem(polygon)
        return polygon
    
    def add_polygon_point(self, point):
        # print("Adding point")
        # self.polygon_points.append(point)
        points_list = [tuple(h.pos()) for h in self.current_polygon.getHandles()]
        # points_list.append((int(point.x()),int(point.y())))
        points_list.append((float(round(point.x())), float(round(point.y(),0))))
        # print("points list: ",points_list,type(points_list))
        self.current_polygon.setPoints(points_list)

    def add_arc(self):
        bounds = QtCore.QRect()
        bounds.setLeft(0)
        bounds.setRight(self.image_data.shape[0])
        bounds.setBottom(0)
        bounds.setTop(self.image_data.shape[1])
        arc = Arc(parent=self.view)
        self.view.addItem(arc, ignoreBounds=True)
        return arc
    
    def set_arc_point(self, point):
        self.current_arc.initialize_handles(point)

    # def clear_polygon(self):
    #     self.polygon_points = []
    #     self.polygon.setPoints(self.polygon_points)

    def mouse_click(self, evt):
        if ((evt.button() == 1) or (evt.button() == pg.QtCore.Qt.MouseButton.LeftButton)) and (self.view.sceneBoundingRect().contains(evt.pos())):
            mousePoint = self.view.vb.mapSceneToView(evt.scenePos())
            # print(mousePoint)
            if self.current_polygon != None:
                # print("polygon")
                # self.add_points_to.addFreeHandle(mousePoint)
                # self.add_points_to.addSegment(self.add_points_to.handles[-2]['item'],self.add_points_to.handles[-1]['item'])

                # points_list = [tuple(h.pos()) for h in self.add_points_to.getHandles()]
                # points_list.append(mousePoint.toPoint())
                # self.add_points_to.setPoints(points_list)
                self.add_polygon_point(mousePoint.toPoint())
            elif self.current_point != None:
                # print("point")
                x = mousePoint.toPoint().x()
                y = mousePoint.toPoint().y()
                self.current_point.setX(x)
                self.current_point.setY(y)
                self.current_point.verifyState()
                self.current_point.table_item.setText("("+str(self.current_point.x())+","+str(self.current_point.y())+")")
            elif self.current_arc != None:
                self.set_arc_point(mousePoint.toPoint())
            # self.add_point(mousePoint)
            # self.add_point(evt.pos())
            # self.qpicture = self.paint_polygon()
            # self.actually_draw_polygon()
            # self.polygon.setPoints(self.polygon_points)

        # elif evt.button() == 2:
        #     # print("Right click")
        #     self.clear_polygon()

    #     self.sendHoverEvents(ev)  ## let items prepare for next click/drag
    def mouseReleaseEvent(self, evt):
        if (evt.button() == 1) or (evt.button() == pg.QtCore.Qt.MouseButton.LeftButton):
            if len(self.objects) > 0:
                for object in self.objects:
                    object.verifyState()
        super().mouseReleaseEvent(evt)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    sys.exit(app.exec())
