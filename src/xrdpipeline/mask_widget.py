import numpy as np
import pyqtgraph as pg
import tifffile as tf
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major")


def choose_file():
    # just tiff for now
    image_file_name = QtWidgets.QFileDialog.getOpenFileName(
        None, "Choose Image", ".", "TIFF files (*.tif)"
    )
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
            [key, val] = S.strip().split(":", 1)
            if key in ["Points", "Rings", "Arcs", "Polygons", "Frames", "Thresholds"]:
                masks[key] = eval(val)
            S = infile.readline()
    for key in ["Points", "Rings", "Arcs", "Polygons"]:
        masks[key] = masks.get(key, [])
        masks[key] = [i for i in masks[key] if len(i)]
    return masks


# class Polygon(pg.GraphicsObject):
#     def __init__(self,points=[]):
#         super().__init__()
#         self.points = points
#         self.picture = None
#         self.paint_polygon()

#     def paint_polygon(self):
#         # TODO: add single point, line for 1-2 points in self.points
#         self.picture = QtGui.QPicture()
#         painter = QtGui.QPainter(self.picture)
#         # painter.setBrush(QtGui.QColor.QColor("cyan"))
#         painter.setBrush(pg.mkColor('c'))

#         painter.drawConvexPolygon(self.points)
#         painter.end()
#         # return picture

#     def paint(self,painter,*args):
#         painter.drawPicture(0,0,self.picture)

#     def boundingRect(self):
#         return QtCore.QRectF(self.picture.boundingRect())

#     def setData(self,points):
#         self.points = points
#         self.paint_polygon()
#         self.update()


class Polygon(pg.PolyLineROI):
    def __init__(self, image_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        # setPoints() is implemented in PolyLineROI
        # add point with:
        # addFreeHandle(point)
        # addSegment(self.handles[-2]['item'],self.handles[-1]['item'])
        # find points with self.handles
        # self.addFreeHandle(QtCore.QPoint(0,0)) # maxBounds are set for only the first point; resets with clearPoints()
        # self.clearPoints()
        self.table_item = QtWidgets.QTableWidgetItem("Polygon")
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

    def verifyState(self):
        # something doesn't properly update when dragging oob, so:
        points = [(h[1].x(), h[1].y()) for h in self.getLocalHandlePositions()]
        print(points)
        self.setPoints(points)


class MainWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
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
        # ROI mask options
        self.add_polygon_button = QtWidgets.QPushButton("New Polygon")
        self.add_polygon_button.released.connect(self.add_polygon)
        self.done_creating_button = QtWidgets.QPushButton("Done Creating")
        self.done_creating_button.released.connect(self.done_creating)
        self.remove_polygon_button = QtWidgets.QPushButton("Delete Selected Polygon")
        self.remove_polygon_button.released.connect(self.delete_selected_polygon)
        # ellipse/circle
        # arc - needs azimuth map

        # tth/q thresholds - needs tth/q map. GSASII has this threshold in the config file.

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
        self.polygons_label = QtWidgets.QLabel("Polygons")
        self.polygons_table = QtWidgets.QTableWidget()
        self.polygons_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.polygons_table.setColumnCount(1)
        self.number_of_objects = 0
        self.polygons_table.setRowCount(self.number_of_objects)
        # self.polygons_table.itemClicked.connect(self.highlightROI)
        # self.polygons_table.selectionModel().currentRowChanged.connect(self.highlightROI)
        self.polygons_table.currentItemChanged.connect(self.highlightROI)

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
        self.object_layout.addWidget(self.polygons_label, 2, 0)
        self.object_layout.addWidget(self.add_polygon_button, 3, 0)
        self.object_layout.addWidget(self.done_creating_button, 3, 1)
        self.object_layout.addWidget(self.polygons_table, 4, 0, 2, 2)
        self.object_layout.addWidget(self.remove_polygon_button, 6, 0)
        self.object_list_widget.setLayout(self.object_layout)
        self.object_list.setWidget(self.object_list_widget)

        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.addWidget(self.main_image, 0, 0, 5, 5)
        self.gridlayout.addWidget(self.load_immask_button, 0, 5)
        self.gridlayout.addWidget(self.save_immask_button, 0, 6)
        self.gridlayout.addWidget(self.object_list, 1, 5, 4, 2)
        self.gridlayout.addWidget(self.preview_mask_button, 5, 5)
        self.gridlayout.addWidget(self.save_mask_button, 5, 6)
        self.setLayout(self.gridlayout)
        self.show()

    def add_polygon(self):
        poly = self.main_image.add_polygon()
        self.number_of_objects += 1
        self.polygons_table.setRowCount(self.number_of_objects)
        self.polygons_table.setItem(self.number_of_objects - 1, 0, poly.table_item)
        self.main_image.objects.append(poly)
        self.main_image.add_points_to = self.main_image.objects[-1]

    def done_creating(self):
        self.main_image.add_points_to = None

    def preview_mask(self):
        if self.preview_mask_button.text() == "Preview Mask":
            self.predef_mask = np.zeros_like(self.main_image.image_data, dtype=bool)
            for i in self.main_image.objects:
                # poly_mask = i.getArrayRegion(np.ones_like(self.main_image.image_data,dtype=bool),self.main_image.image, returnMappedCoords=True)
                # print(poly_mask, poly_mask[0].shape)
                # print(poly_mask.pos()) # should be lower-left corner
                # poly_mask = i.renderShapeMask(2880,2880) # mask is drawn to size given
                # Play with transpose because this doesn't recognize row-major images, but still knows positions by x,y
                print(i.getLocalHandlePositions())
                print([tuple(h.pos()) for h in i.getHandles()])
                print([(h[1].x(), h[1].y()) for h in i.getLocalHandlePositions()])
                points = [(h[1].x(), h[1].y()) for h in i.getLocalHandlePositions()]
                # i.clearPoints()
                corrected_points = points
                for it in range(len(corrected_points)):
                    print(corrected_points[it])
                    if corrected_points[it][0] >= self.main_image.image_data.shape[0]:
                        corrected_points[it] = (
                            self.main_image.image_data.shape[0] - 1,
                            corrected_points[it][1],
                        )
                    if corrected_points[it][1] >= self.main_image.image_data.shape[1]:
                        corrected_points[it] = (
                            corrected_points[it][0],
                            self.main_image.image_data.shape[1] - 1,
                        )
                    print(corrected_points[it])
                print("Corrected points:", corrected_points)
                i.setPoints(corrected_points)
                # i.stateChanged()
                array_slice = i.getArraySlice(
                    self.main_image.image_data, self.main_image.image, returnSlice=False
                )
                print(array_slice)
                print(
                    np.abs(array_slice[0][1][0] - array_slice[0][1][1]),
                    np.abs(array_slice[0][0][0] - array_slice[0][0][1]),
                )
                small_poly_mask = i.renderShapeMask(
                    np.abs(array_slice[0][1][0] - array_slice[0][1][1]),
                    np.abs(array_slice[0][0][0] - array_slice[0][0][1]),
                )
                # small_poly_mask = i.renderShapeMask(np.abs(array_slice[0][0][0]-array_slice[0][0][1]),np.abs(array_slice[0][1][0]-array_slice[0][1][1]))
                poly_mask = np.zeros_like(self.main_image.image_data, dtype=bool)
                array_slice = i.getArraySlice(
                    self.main_image.image_data, self.main_image.image
                )
                print(array_slice)
                print(array_slice[0][1], array_slice[0][0])
                # poly_mask[array_slice[0]] = small_poly_mask
                poly_mask[(array_slice[0][1], array_slice[0][0])] = small_poly_mask
                self.predef_mask = np.logical_or(self.predef_mask, poly_mask.T)
                i.setPoints(points)
                # print(i.boundingRect(),i.parentBounds(),i.pos())

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
        tf.imwrite("predef_mask.tif", self.predef_mask)

    def save_immask(self):
        # GSASII outputs/expects an explicit return to the first point
        points_list = []
        for i in self.main_image.objects:
            points = i.saveState()["points"]
            points.append(points[0])
            points_list.append(points)
        # TODO: choice of outfile location
        outfilename = "./mask.immask"
        with open(outfilename, "w") as outfile:
            outfile.write("Points:[]\n")
            outfile.write("Rings:[]\n")
            outfile.write("Arcs:[]\n")
            print(points_list)
            outfile.write(
                "Polygons:{polys}\n".format(polys=points_list)
                .replace("(", "[")
                .replace(")", "]")
            )
            outfile.write("Xlines:[]\n")
            outfile.write("Ylines:[]\n")
            outfile.write("Frames:[]\n")
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
        # Polygons only for now
        for poly in masks["Polygons"]:
            self.add_polygon()
            for point in poly:
                self.main_image.add_point(QtCore.QPoint(int(point[0]), int(point[1])))
        self.done_creating()

    # def mouseReleaseEvent(self,evt):
    #     print("on mouse release")
    #     if (evt.button() == 1):
    #         if len(self.objects) > 0:
    #             for object in self.objects:
    #                 object.verifyState()
    #     super().mouseReleaseEvent(evt)

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
        # ~poly.table_item
        # resize table, shift all up # looks like it shifts things up by itself
        max_index = self.number_of_objects
        # for i in range(index,max_index):
        #     self.polygons_table.setItem(i,0,self.main_image.objects[i+1].table_item)
        self.polygons_table.setRowCount(self.number_of_objects)
        del max_index
        # remove from objects list
        del self.main_image.objects[index]

    def delete_selected_polygon(self):
        selected = self.polygons_table.selectedItems()[
            0
        ]  # list of singular item, as selection mode is single selection
        self.clear_polygon(index=selected.row())

    def highlightROI(self, current, previous):
        # print(current.row())
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
        self.image_data = tf.imread(choose_file()[0])
        self.image = pg.ImageItem(self.image_data)
        self.predef_mask_data = np.zeros(
            (self.image_data.shape[0], self.image_data.shape[1], 4), dtype=np.uint8
        )
        self.predef_mask = pg.ImageItem(self.predef_mask_data, levels=None)
        # self.qpicture = None
        self.add_points_to = None
        self.objects = []

        self.view.addItem(self.image)
        # self.view.setRange(xRange=(0,self.image_data.shape[0]),yRange=(0,self.image_data.shape[1]))
        # self.view.vb.setLimits(xMin=0,xMax=self.image_data.shape[0],yMin=0,yMax=self.image_data.shape[1])
        print(self.view.viewRange())
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

    def add_polygon(self):
        # bounds = QtCore.QRect(0,2880,2880,2880)
        # bounds = QtCore.QRect(QtCore.QPoint(2880,0),QtCore.QPoint(0,2880))
        bounds = QtCore.QRect()
        bounds.setLeft(0)
        bounds.setRight(2880)
        bounds.setBottom(0)
        bounds.setTop(2880)
        print(
            bounds,
            bounds.bottomLeft(),
            bounds.bottomRight(),
            bounds.topLeft(),
            bounds.topRight(),
        )
        # y goes from 0 to 2880, but x goes from 1 to 2879?
        print(self.image_data.shape)  # 2880,2880
        print(self.image.boundingRect())
        # need to transform bounding rect back to proper coords
        # polygon = Polygon(positions=[],closed=True,translateSnap=True,rotatable=False,maxBounds=self.image.boundingRect())
        polygon = Polygon(
            positions=[],
            closed=True,
            translateSnap=True,
            rotatable=False,
            image_size=self.image_data.shape,
            parent=self.view,
        )

        self.view.addItem(polygon)
        return polygon

    def add_point(self, point):
        # print("Adding point")
        # self.polygon_points.append(point)
        points_list = [tuple(h.pos()) for h in self.add_points_to.getHandles()]
        points_list.append((int(point.x()), int(point.y())))
        print("points list: ", points_list, type(points_list))
        self.add_points_to.setPoints(points_list)

    # def clear_polygon(self):
    #     self.polygon_points = []
    #     self.polygon.setPoints(self.polygon_points)

    def mouse_click(self, evt):
        if (evt.button() == 1) and (self.view.sceneBoundingRect().contains(evt.pos())):
            mousePoint = self.view.vb.mapSceneToView(evt.scenePos())
            print(mousePoint)
            if self.add_points_to != None:
                # self.add_points_to.addFreeHandle(mousePoint)
                # self.add_points_to.addSegment(self.add_points_to.handles[-2]['item'],self.add_points_to.handles[-1]['item'])

                # points_list = [tuple(h.pos()) for h in self.add_points_to.getHandles()]
                # points_list.append(mousePoint.toPoint())
                # self.add_points_to.setPoints(points_list)
                self.add_point(mousePoint.toPoint())
            # self.add_point(mousePoint)
            # self.add_point(evt.pos())
            # self.qpicture = self.paint_polygon()
            # self.actually_draw_polygon()
            # self.polygon.setPoints(self.polygon_points)

        # elif evt.button() == 2:
        #     # print("Right click")
        #     self.clear_polygon()

    # def mouse_release(self,evt):
    #     if (evt.button() == 1):
    #         if self.add_points_to != None:
    #             self.add_points_to.verifyState()

    # Reimplementing GraphicsScene's mouseReleaseEvent to avoid loss of functionality:
    # def mouseReleaseEvent(self, ev):
    #     if self.mouseGrabberItem() is None:
    #         if ev.button() in self.dragButtons:
    #             if self.sendDragEvent(ev, final=True):
    #                 #print "sent drag event"
    #                 ev.accept()
    #             self.dragButtons.remove(ev.button())
    #         else:
    #             cev = [e for e in self.clickEvents if e.button() == ev.button()]
    #             if cev:
    #                 if self.sendClickEvent(cev[0]):
    #                     ev.accept()
    #                 try:
    #                     self.clickEvents.remove(cev[0])
    #                 except ValueError:
    #                     warnings.warn(
    #                         ("A ValueError can occur here with errant "
    #                          "QApplication.processEvent() calls, see "
    #                         "https://github.com/pyqtgraph/pyqtgraph/pull/2580 "
    #                         "for more information."),
    #                         RuntimeWarning,
    #                         stacklevel=2
    #                     )
    #     if not ev.buttons():
    #         self.dragItem = None
    #         self.dragButtons = []
    #         self.clickEvents = []
    #         self.lastDrag = None
    #     super().mouseReleaseEvent(ev)

    #     self.sendHoverEvents(ev)  ## let items prepare for next click/drag
    def mouseReleaseEvent(self, evt):
        print("on mouse release")
        if evt.button() == 1:
            if len(self.objects) > 0:
                for object in self.objects:
                    object.verifyState()
        super().mouseReleaseEvent(evt)


if __name__ == "__main__":
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication([])
    # main_image = MainImage()
    main_window = MainWindow()
    app.exec_()
