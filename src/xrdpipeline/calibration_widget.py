import os
import sys
import html
from itertools import combinations

# Select the same Qt binding used by mask_widget.py.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

import numpy as np
import tifffile as tf
import pyqtgraph as pg

from PySide6 import QtCore, QtWidgets
from scipy.optimize import brentq, least_squares
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

GEOMETRY_KEYS = (
    "Beam Center X (px) ",
    "Beam Center Y (px) ",
    "Distance (mm) ",
    "Tilt (deg) ",
    "Tilt Rotation (deg) ",
    "Penetration (fraction) ",
)

# Wavelength participates in fitting but is not a geometry coordinate.
WAVELENGTH_KEY = "Wavelength (Å) "
FIT_KEYS = GEOMETRY_KEYS + (WAVELENGTH_KEY,)

pg.setConfigOptions(imageAxisOrder="row-major")

# ==============================================================================
# GSAS-II IMAGE CALIBRANT DICTIONARY LOOKUP ENGINE
# ==============================================================================
# Mirrors the default constants found within GSAS-II's ImageCalibrants.py module.
CALIBRANTS = {
    'Silicon (SRM 640c)': {
        'lattice': 5.4311946, 'system': 'Cubic',
        'hkl': [[1,1,1], [2,2,0], [3,1,1], [4,0,0], [3,3,1], [4,2,2], [5,1,1], [4,4,0]]
    },
    'CeO2 (NIST SRM 674b)': {
        'lattice': 5.411651, 'system': 'Cubic',
        'hkl': [[1,1,1], [2,0,0], [2,2,0], [3,1,1], [2,2,2], [4,0,0], [3,3,1], [4,2,0]]
    },
    'Ni (@ 298 K)': {
        'lattice': 3.52475, 'system': 'Cubic',
        'hkl': [[1,1,1], [2,0,0], [2,2,0], [3,1,1], [2,2,2], [4,0,0], [3,3,1], [4,2,0]]
    },    
    'LaB6 (NIST SRM 660a)': {
        'lattice': 4.1569162, 'system': 'Cubic',
        'hkl': [[1,0,0], [1,1,0], [1,1,1], [2,0,0], [2,1,0], [2,1,1], [2,2,0], [3,0,0]]
    },
    'Silver Behenate': {
        'lattice': 58.38, 'system': 'Triclinic/Layered',
        'hkl': [[0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6], [0,0,7], [0,0,8]]
    }
}

def calculate_d_spacings(name, wavelength):
    """Calculates ideal 2-Theta scattering positions based on crystal metrics."""
    cal = CALIBRANTS[name]
    a = cal['lattice']
    d_list = []
    tth_list = []
    
    for hkl in cal['hkl']:
        h, k, l = hkl
        if cal['system'] == 'Cubic':
            d = a / np.sqrt(h**2 + k**2 + l**2)
        else: # Layered standard shortcut (Silver Behenate (00l) tracks)
            d = a / l
        d_list.append(d)
        # Bragg's Law: lambda = 2d sin(theta) -> theta = arcsin(lambda/(2d))
        val = wavelength / (2.0 * d)
        if val < 1.0:
            tth_list.append(2.0 * np.arcsin(val))
    return np.array(tth_list)

# ==============================================================================
# GEOMETRIC TRANSFORMATION MATHEMATICAL ENGINE (GSAS-II FORWARD PHYSICS)
# ==============================================================================

def gsas2_penetration_correction(two_theta, penetration, distance):
    """GSAS-II-style empirical penetration correction.

    Parameters
    ----------
    two_theta : array_like
        Uncorrected scattering angle, in radians.
    penetration : float
        Dimensionless DetDepth parameter.
    distance : float
        Distance used by the current geometric model, in mm.

    Returns
    -------
    ndarray
        Longitudinal correction in mm.
    """
    two_theta = np.asarray(two_theta, dtype=float)

    # Equivalent to penetration * distance * (1 - cos(two_theta)),
    # but numerically more stable at small angles.
    return (
        penetration
        * distance
        * 2.0
        * np.sin(0.5 * two_theta) ** 2
    )

def gsas2_metric_projection(pts_x, pts_y, x0, y0, distance, tilt, tilt_rotation, pixel_size_x, pixel_size_y, penetration=0.0):
    """
    Transforms pixel positions into a true 3D spatial geometry projection.
    Convert displayed detector coordinates to scattering angles.
    Returns corrected 2-theta angles in radians.
    
    Parameters:
      pts_x, pts_y : Arrays of coordinates extracted directly from pixel positions.
      x0, y0       : Orthogonal baseline beam intercept coordinates on the detector face (mm).
      distance     : Perpendicular linear distance from sample to detector plane (mm).
      tilt         : Total misalignment angle of the detector face plane (degrees).
      tilt_rotation : GSAS-II-style rotation-axis orientation, in degrees. Tilt azimuth is rotation -90 degree
    pixel_size_x : Horizontal pixel size (mm).
    pixel_size_y : Vertical pixel size (mm).
      penetration is the dimensionless empirical DetDepth parameter.
    """
    # Convert pixels to physical mm space relative to origin (top-left)
    x_mm = pts_x * pixel_size_x
    y_mm = pts_y * pixel_size_y
    
    # Recenter coordinates relative to the beam center projection plane
    dx = x_mm - x0
    dy = y_mm - y0
    
    # Convert angular variables to radians
    t_rad = np.radians(tilt)
    p_rad = np.radians(tilt_rotation - 90.0)
    
    # GSAS-II Vector Rotations: Rotate coordinates into tilted detector plane
    # Projects the local offset back onto the real X-ray path trajectory
    cos_p, sin_p = np.cos(p_rad), np.sin(p_rad)
    cos_t, sin_t = np.cos(t_rad), np.sin(t_rad)
    
    # Component transformations alongside the physical azimuth orientation vector
    x_rot =  dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p
    
    transverse = np.hypot(y_rot, x_rot * cos_t)
    longitudinal = distance - x_rot * sin_t

    uncorrected = np.arctan2(transverse, longitudinal)

    delta_z = gsas2_penetration_correction(
        uncorrected,
        penetration,
        distance,
    )

    return np.arctan2(
        transverse,
        longitudinal + delta_z,
    )

# ==============================================================================
# MAIN WINDOW USER INTERFACE ECOSYSTEM
# ==============================================================================
class GSAS2CalibratorApp(QtWidgets.QWidget):
    def __init__(self, pipeline_handoff_file=None):
        super().__init__()

        self.pipeline_handoff_file = pipeline_handoff_file

        self.setWindowTitle("2D Diffraction Calibrator")
        self.resize(1300, 850)
        self.setMinimumSize(1000, 600)

        self.image_data = None
        self.image_file_name = None
        self.calibrated_geometry = None

        self.tracked_points_x = []
        self.tracked_points_y = []
        self.pixel_size_x = 0.2
        self.pixel_size_y = 0.2
        self.pick_beam_center_mode = False

        # Each entry stores coordinates and the assigned reflection.
        self.tracked_rings = []

        self.init_ui()

    def init_ui(self):
        # ==============================================================
        # IMAGE DISPLAY: pyqtgraph, like mask_widget.py
        # ==============================================================

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")

        self.view = self.graphics.addPlot(row=0, col=0)
        self.view.setAspectLocked(True)
        self.view.setLabel("bottom", "X", units="px")
        self.view.setLabel("left", "Y", units="px")
        self.view.setTitle("Load a TIFF diffraction image")

        # Row-major indexing:
        # image_data[y, x] corresponds to displayed coordinate (x, y).
        self.image = pg.ImageItem()
        self.view.addItem(self.image)

        self.intensity_bar = pg.HistogramLUTItem()
        self.intensity_bar.setImageItem(self.image)
        self.intensity_bar.gradient.loadPreset("grey")
        self.intensity_bar.gradient.showTicks(show=False)
        self.graphics.addItem(self.intensity_bar, row=0, col=1)

        self.coord_label = pg.LabelItem(justify="left", color="k")
        self.graphics.addItem(
            self.coord_label, row=1, col=0, colspan=2
        )

        self.tracked_scatter = pg.ScatterPlotItem(
            size=4,
            pen=None,
            brush=pg.mkBrush("cyan"),
            pxMode=True,
        )
        self.view.addItem(self.tracked_scatter)

        self.spoke_pick_scatter = pg.ScatterPlotItem(
            symbol="o",
            size=9,
            pen=pg.mkPen("#202020", width=1),
            brush=pg.mkBrush("#ffe45c"),
            pxMode=True,
        )
        self.spoke_pick_scatter.setZValue(4)
        self.view.addItem(self.spoke_pick_scatter)
        
        # Two short red lines mark the calibrated beam center.
        marker_pen = pg.mkPen("red", width=2)
        self.beam_center_horizontal = pg.PlotDataItem(
            pen=marker_pen,
            connect="all",
        )
        self.beam_center_vertical = pg.PlotDataItem(
            pen=marker_pen,
            connect="all",
        )
        for marker in (
            self.beam_center_horizontal,
            self.beam_center_vertical,
        ):
            marker.setZValue(3)
            marker.hide()
            self.view.addItem(marker)

        self.inner_curve = pg.PlotDataItem(
            pen=pg.mkPen(
                "red",
                width=1.5,
                style=QtCore.Qt.PenStyle.DashLine,
            )
        )
        self.outer_curve = pg.PlotDataItem(
            pen=pg.mkPen(
                "#ff9800",
                width=1.5,
                style=QtCore.Qt.PenStyle.DashLine,
            )
        )

        self.view.addItem(self.inner_curve)
        self.view.addItem(self.outer_curve)

        integration_pens = (
            pg.mkPen("#ad1457", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            pg.mkPen("#1565c0", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            pg.mkPen("#2e7d32", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            pg.mkPen("#2e7d32", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.integration_limit_items = [
            pg.PlotDataItem(pen=pen) for pen in integration_pens
        ]
        for item in self.integration_limit_items:
            item.setZValue(1.5)
            item.hide()
            self.view.addItem(item)

        # Ensure overlays appear above the image.
        self.image.setZValue(0)
        self.inner_curve.setZValue(1)
        self.outer_curve.setZValue(1)
        self.tracked_scatter.setZValue(2)

        self.view.scene().sigMouseClicked.connect(
            self.handle_mouse_click_pick
        )
        self.view.scene().sigMouseMoved.connect(self.mouse_moved)

        # ==============================================================
        # RIGHT CONTROL PANEL
        # ==============================================================

        panel = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(panel)

        self.btn_help = QtWidgets.QPushButton("Quick Guide")
        self.btn_help.clicked.connect(self.show_help)

        self.btn_load_image = QtWidgets.QPushButton("Load image")
        self.btn_load_image.clicked.connect(self.load_image_popup)

        image_guide_row = QtWidgets.QWidget()
        image_guide_layout = QtWidgets.QHBoxLayout(image_guide_row)
        image_guide_layout.setContentsMargins(0, 0, 0, 0)
        image_guide_layout.addWidget(self.btn_load_image)
        image_guide_layout.addWidget(self.btn_help)
        controls.addWidget(image_guide_row)

        self.lbl_file = QtWidgets.QLabel("No TIFF image loaded.")
        self.lbl_file.setWordWrap(True)
        controls.addWidget(self.lbl_file)

        # Hardware specifications.
        hw_box = QtWidgets.QGroupBox("Detector Specifications")
        hw_grid = QtWidgets.QGridLayout(hw_box)

        self.txt_pixel_x = QtWidgets.QLineEdit("0.2")
        self.txt_pixel_y = QtWidgets.QLineEdit("0.2")

        hw_grid.addWidget(
            QtWidgets.QLabel("Pixel X (mm):"), 0, 0
        )
        hw_grid.addWidget(self.txt_pixel_x, 0, 1)
        hw_grid.addWidget(
            QtWidgets.QLabel("Pixel Y (mm):"), 0, 2
        )
        hw_grid.addWidget(self.txt_pixel_y, 0, 3)

        controls.addWidget(hw_box)

        # Calibrant library.
        cal_box = QtWidgets.QGroupBox(
            "Calibrant Reference"
        )
        cal_layout = QtWidgets.QGridLayout(cal_box)

        self.combo_calibrant = QtWidgets.QComboBox()
        self.combo_calibrant.addItems(list(CALIBRANTS.keys()))

        cal_layout.addWidget(self.combo_calibrant, 0, 0, 1, 4)
        self.txt_lattice = QtWidgets.QLineEdit()
        cal_layout.addWidget(
            QtWidgets.QLabel("Lattice (Å):"), 1, 0
        )
        cal_layout.addWidget(self.txt_lattice, 1, 1)
        cal_layout.addWidget(
            QtWidgets.QLabel("Min d (Å):"), 1, 2
        )
        self.txt_min_d = QtWidgets.QLineEdit("1.0")
        cal_layout.addWidget(self.txt_min_d, 1, 3)

        self.combo_reflection = QtWidgets.QComboBox()
        cal_layout.addWidget(
            QtWidgets.QLabel("Ring-to-trace:"), 2, 0
        )
        cal_layout.addWidget(self.combo_reflection, 2, 1, 1, 3)

        self.lbl_ring_summary = QtWidgets.QLabel("No rings collected.")
        self.lbl_ring_summary.setWordWrap(True)

        self.combo_calibrant.currentTextChanged.connect(
            self.update_calibrant_controls
        )
        self.update_calibrant_controls()
        controls.addWidget(cal_box)

        # Parameter names retain their trailing spaces to match the
        # keys used by your existing refinement method.
        param_box = QtWidgets.QGroupBox(
            "Calibration Parameters"
        )
        self.grid_param = QtWidgets.QGridLayout(param_box)

        self.inputs = {}

        defaults = [
            ("Beam Center X (px) ", "250.0", True),
            ("Beam Center Y (px) ", "250.0", True),
            ("Distance (mm) ", "200.0", True),
            ("Tilt (deg) ", "0.0", True),
            ("Tilt Rotation (deg) ", "0.0", True),
            ("Penetration (fraction) ", "0.0", False),
            (WAVELENGTH_KEY, "0.2099", False),
        ]

        for row, (key, value, refine) in enumerate(defaults):
            edit = QtWidgets.QLineEdit(value)
            checkbox = QtWidgets.QCheckBox("Refine")
            checkbox.setChecked(refine)

            self.grid_param.addWidget(
                QtWidgets.QLabel(key.strip()), row, 0
            )
            self.grid_param.addWidget(edit, row, 1)
            self.grid_param.addWidget(checkbox, row, 2)

            self.inputs[key] = (edit, checkbox)
            edit.editingFinished.connect(self.plot_image)

        # Preserve this name because expansion and other existing methods use it.
        self.txt_wave = self.inputs[WAVELENGTH_KEY][0]

        self.txt_wave.setToolTip(
            "X-ray wavelength in Å. Refine using a known calibrant lattice "
            "parameter. Wavelength can be strongly correlated with distance."
        )
        
        self.inputs[WAVELENGTH_KEY][1].setToolTip(
            "Unchecked by default. For low-angle data, consider fixing "
            "distance while refining wavelength."
        )

        depth_edit, depth_refine = self.inputs["Penetration (fraction) "]

        depth_edit.setToolTip(
            "GSAS-II-style DetDepth: a dimensionless empirical penetration "
            "parameter, not detector thickness in mm. "
            "Zero disables the correction."
        )

        depth_refine.setToolTip(
            "Refine penetration only after collecting several rings over "
            "a useful angular range. It can be strongly correlated with distance."
        )

        controls.addWidget(param_box)

        # Search boundaries.
        edge_box = QtWidgets.QGroupBox(
            "Auto Calibrate Search boundaries"
        )
        edge_grid = QtWidgets.QGridLayout(edge_box)

        self.txt_inner_edge = QtWidgets.QLineEdit("50")
        self.txt_outer_edge = QtWidgets.QLineEdit("230")

        edge_grid.addWidget(
            QtWidgets.QLabel("Inner Radius (px):"), 0, 0
        )
        edge_grid.addWidget(self.txt_inner_edge, 0, 1)

        edge_grid.addWidget(
            QtWidgets.QLabel("Outer Radius (px):"), 0, 2
        )
        edge_grid.addWidget(self.txt_outer_edge, 0, 3)

        self.chk_show_edges = QtWidgets.QCheckBox(
            "Plot Boundary Edges"
        )
        self.chk_show_edges.toggled.connect(self.plot_image)
        edge_grid.addWidget(self.chk_show_edges, 1, 0, 1, 4)

        self.txt_inner_edge.editingFinished.connect(self.plot_image)
        self.txt_outer_edge.editingFinished.connect(self.plot_image)

        controls.addWidget(edge_box)
        #edge_box.hide()

        # Use standard Qt button styling, as in mask_widget.py.
        self.btn_pick_center = QtWidgets.QPushButton("Pick Beam Center")
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setToolTip(
            "Click this, then click the estimated beam center in the image."
        )
        self.btn_pick_center.toggled.connect(
            self.toggle_beam_center_pick
        )

        self.btn_spoke_calibrate = QtWidgets.QPushButton(
            "8-Spoke Auto Calibrate"
        )
        self.btn_spoke_calibrate.setToolTip(
            "Detect the first ring along eight radial directions, robustly "
            "fit its center, then calibrate and expand the rings."
        )
        self.btn_spoke_calibrate.clicked.connect(
            self.run_eight_spoke_calibration
        )

        self.btn_refine = QtWidgets.QPushButton(
            "Calibrate"
        )
        self.btn_refine.clicked.connect(
            self.run_gsas2_optimization_refinement
        )

        calibration_button_row = QtWidgets.QWidget()
        calibration_button_layout = QtWidgets.QHBoxLayout(
            calibration_button_row
        )
        calibration_button_layout.setContentsMargins(0, 0, 0, 0)
        calibration_button_layout.addWidget(self.btn_spoke_calibrate)
        calibration_button_layout.addWidget(self.btn_refine)

        self.btn_reoptimize = QtWidgets.QPushButton(
            "Recalibrate"
        )
        self.btn_reoptimize.setToolTip(
            "Resample every stored reflection using the current geometry, "
            "then refit all rings."
        )
        self.btn_reoptimize.clicked.connect(
            self.reoptimize_current_rings
        )

        self.btn_clear = QtWidgets.QPushButton(
            "Clear Tracked Points"
        )
        self.btn_clear.clicked.connect(self.clear_tracked_nodes)

        self.btn_export_imctrl = QtWidgets.QPushButton(
            "Save GSAS-II .imctrl"
        )
        self.btn_export_imctrl.setToolTip(
            "Save the calibrated geometry in GSAS-II image-control format."
        )
        self.btn_export_imctrl.clicked.connect(
            self.export_calibration_to_imctrl
        )

        self.btn_save_pipeline = QtWidgets.QPushButton(
            "Save and Export to Pipeline"
        )
        self.btn_save_pipeline.setToolTip(
            "Save a GSAS-II image control file, close this window, "
            "and load that file in the pipeline."
        )
        self.btn_save_pipeline.clicked.connect(
            self.save_and_export_to_pipeline
        )

        controls.addWidget(self.btn_pick_center)
        controls.addWidget(calibration_button_row)

        for button in (
            self.btn_reoptimize,
            self.btn_clear,
            self.btn_export_imctrl,
            self.btn_save_pipeline,
        ):
            controls.addWidget(button)
            button.setEnabled(False)

        self.btn_pick_center.setEnabled(False)
        self.btn_spoke_calibrate.setEnabled(False)
        self.btn_refine.setEnabled(False)

        '''
        help_label = QtWidgets.QLabel(
            "Load a TIFF image, select the calibrant, and set the "
            "pixel size and wavelength.\n"
            "Click once on the first ring to begin ring tracing.\n"
        )
        help_label.setWordWrap(True)
        controls.addWidget(help_label)
        '''
        self.lbl_status = QtWidgets.QLabel(
            "Status: Load a TIFF image to begin."
        )
        self.lbl_status.setWordWrap(True)

        trace_box = QtWidgets.QGroupBox("Ring Peak-Picking Settings")
        trace_form = QtWidgets.QFormLayout(trace_box)

        def double_setting(value, minimum, maximum, step, decimals=2):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(decimals)
            widget.setSingleStep(step)
            widget.setValue(value)
            return widget

        self.spin_search_width = double_setting(
            3.0, 0.5, 30.0, 0.5
        )
        self.spin_search_width.setToolTip(
            "Radial search half-width. A value of 3 searches "
            "from predicted radius - 3 to predicted radius + 3 pixels."
        )
        trace_form.addRow(
            "Search half-width (px):", self.spin_search_width
        )

        self.spin_radial_step = double_setting(
            0.25, 0.05, 1.0, 0.05
        )
        trace_form.addRow(
            "Radial sampling step (px):", self.spin_radial_step
        )

        self.spin_smooth_sigma = double_setting(
            0.25, 0.0, 3.0, 0.05
        )
        self.spin_smooth_sigma.setToolTip(
            "Gaussian smoothing width in physical pixels, "
            "not in radial sample counts. Zero disables smoothing."
        )
        trace_form.addRow(
            "Smoothing sigma (px):", self.spin_smooth_sigma
        )

        self.spin_prominence_fraction = double_setting(
            0.15, 0.0, 1.0, 0.05
        )
        trace_form.addRow(
            "Minimum prominence fraction:",
            self.spin_prominence_fraction,
        )

        self.spin_noise_multiplier = double_setting(
            4.0, 0.0, 20.0, 0.5
        )
        trace_form.addRow(
            "Noise threshold multiplier:", self.spin_noise_multiplier
        )

        self.spin_peak_separation = double_setting(
            1.0, 0.1, 10.0, 0.1
        )
        trace_form.addRow(
            "Minimum peak separation (px):",
            self.spin_peak_separation,
        )

        self.spin_trace_angles = QtWidgets.QSpinBox()
        self.spin_trace_angles.setRange(36, 1440)
        self.spin_trace_angles.setSingleStep(36)
        self.spin_trace_angles.setValue(360)
        trace_form.addRow(
            "Azimuth samples:", self.spin_trace_angles
        )

        self.chk_pixel_centers = QtWidgets.QCheckBox(
            "Correct pyqtgraph pixel-center coordinates"
        )
        self.chk_pixel_centers.setChecked(True)
        self.chk_pixel_centers.setToolTip(
            "Subtract 0.5 from displayed X/Y coordinates before "
            "SciPy interpolation. Appropriate for the default, "
            "untransformed pyqtgraph ImageItem."
        )
        trace_form.addRow(self.chk_pixel_centers)

        self.chk_show_trace_settings = QtWidgets.QCheckBox(
            "Show ring peak-picking settings"
        )
        self.chk_show_trace_settings.toggled.connect(
            trace_box.setVisible
        )
        trace_box.hide()

        integration_box = QtWidgets.QGroupBox("Integration Settings")
        integration_form = QtWidgets.QFormLayout(integration_box)

        self.spin_inner_tth = double_setting(0.8, 0.0, 180.0, 0.1)
        self.spin_outer_tth = double_setting(10.8, 0.0, 180.0, 0.1)
        self.spin_azimuth_start = double_setting(0.0, 0.0, 360.0, 1.0)
        self.spin_azimuth_end = double_setting(360.0, 0.0, 360.0, 1.0)
        self.spin_azimuth_start.setWrapping(True)
        self.spin_azimuth_end.setWrapping(True)

        integration_form.addRow(
            "Inner 2-theta (deg):", self.spin_inner_tth
        )
        integration_form.addRow(
            "Outer 2-theta (deg):", self.spin_outer_tth
        )
        integration_form.addRow(
            "Start azimuth (deg):", self.spin_azimuth_start
        )
        integration_form.addRow(
            "End azimuth (deg):", self.spin_azimuth_end
        )

        self.spin_integration_bins = QtWidgets.QSpinBox()
        self.spin_integration_bins.setRange(1, 1000000)
        self.spin_integration_bins.setSingleStep(100)
        self.spin_integration_bins.setValue(2000)
        integration_form.addRow(
            "Number of bins:", self.spin_integration_bins
        )

        self.chk_show_integration_limits = QtWidgets.QCheckBox(
            "Show integration limits"
        )
        integration_form.addRow(self.chk_show_integration_limits)
        for widget in (
            self.spin_inner_tth,
            self.spin_outer_tth,
            self.spin_azimuth_start,
            self.spin_azimuth_end,
            self.spin_integration_bins,
        ):
            widget.valueChanged.connect(self.plot_image)
        self.chk_show_integration_limits.toggled.connect(self.plot_image)
        controls.addWidget(integration_box)

        controls.addWidget(self.chk_show_trace_settings)
        controls.addWidget(trace_box)

        controls.addStretch()

        # Scrollable controls, similar to the mask widget.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(380)

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.lbl_status, 0, 0, 1, 2)
        layout.addWidget(self.graphics, 1, 0)
        layout.addWidget(scroll, 1, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)

    def update_calibrant_controls(self, *_):
        """Populate the reflection selector for the chosen standard."""
        name = self.combo_calibrant.currentText()
        cal = CALIBRANTS[name]

        self.txt_lattice.setText(f"{cal['lattice']:.6g}")
        self.combo_reflection.clear()

        for hkl in cal["hkl"]:
            h, k, l = hkl
            self.combo_reflection.addItem(
                f"({h} {k} {l})",
                tuple(hkl),
            )

    def update_ring_summary(self):
        if self.calibrated_geometry is not None:
            count = len(self.tracked_rings)
            label = "ring" if count == 1 else "rings"
            self.lbl_ring_summary.setText(
                f"{count} {label} collected"
            )
            return

        lines = []

        for index, ring in enumerate(self.tracked_rings, start=1):
            h, k, l = ring["hkl"]
            lines.append(
                f"Ring {index}: ({h} {k} {l}), "
                f"{len(ring['x'])} points"
            )

        self.lbl_ring_summary.setText(
            "\n".join(lines) if lines else "No rings collected."
        )

    def set_error_status(self, message):
        escaped_message = html.escape(message)
        styled_message = escaped_message.replace(
            "Error:",
            '<span style="color: red;">Error</span>:',
            1,
        )
        self.lbl_status.setText(styled_message)

    def show_help(self):
        QtWidgets.QMessageBox.information(
            self,
            "Quick Guide",
            "1. Load TIFF image of a calibrant.\n" \
            "2. Set pixel size, calibrant and wavelength. Do not refine wavelength.\n" \
            "3. Click 'Pick Beam Center', then on the image click on a spot inside "
            "the first diffraction ring. Skip this step if the beam center is already inside.\n" \
            "4. Click '8-spoke Auto Calibrate'.\n"
            "5. Click 'Recalibrate' to make sure refined values converge.\n"
            "6. Edit numbers in integration settings (optional step).\n" \
            "7. Save results to imctrl file and/or export to XRDdatapipeline.\n\n" \
            "If step 4 fails, try picking a different spot inside "
            "the first ring for beam center and repeat. Alternative, try manual calibration.\n\n"
            "Manual calibration:\n" \
            "1. Click 'Pick Beam Center' and place it approximately the real beam center.\n" \
            "2. On the image, click anywhere on the first ring. You may zoom in to click more accurately. "
            "Cyan dots will appear indicating the traced points. "
            "If the traced arc is less than 120 degrees, click on the opposite side of the ring to sample more points.\n" \
            "3. Click 'Calibrate'. Then follow steps 5 to 7 in the auto-calibration procedure.\n\n"
            
            )

    def store_traced_ring(self, points_x, points_y):
        """Append one traced ring with its explicit reflection assignment."""
        if not points_x:
            return

        hkl = self.combo_reflection.currentData()
        if hkl is None:
            self.set_error_status("Error: Select a reflection first.")
            return

        calibrant = self.combo_calibrant.currentText()
        points_x = np.asarray(points_x, dtype=float)
        points_y = np.asarray(points_y, dtype=float)
        ring = next(
            (
                candidate
                for candidate in self.tracked_rings
                if candidate["calibrant"] == calibrant
                and candidate["hkl"] == tuple(hkl)
            ),
            None,
        )

        if ring is None:
            self.tracked_rings.append({
                "calibrant": calibrant,
                "hkl": tuple(hkl),
                "x": points_x,
                "y": points_y,
            })
        else:
            ring["x"] = np.concatenate((ring["x"], points_x))
            ring["y"] = np.concatenate((ring["y"], points_y))

        # Preserve the existing combined scatter overlay.
        self.tracked_points_x = np.concatenate(
            [ring["x"] for ring in self.tracked_rings]
        ).tolist()
        self.tracked_points_y = np.concatenate(
            [ring["y"] for ring in self.tracked_rings]
        ).tolist()

        self.update_ring_summary()

    # ==================================================================
    # TIFF LOADING
    # ==================================================================

    def load_image_popup(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose Test Image",
            "",
            "TIFF files (*.tif *.tiff);;All files (*)",
        )

        if filename:
            self.load_image(filename)

    def load_image(self, filename):
        try:
            data = np.asarray(tf.imread(filename))

            # Remove singleton dimensions, but do not silently choose
            # a frame from a stack or convert an RGB image.
            if data.ndim != 2:
                data = np.squeeze(data)

            if data.ndim != 2:
                raise ValueError(
                    "Select a single 2D grayscale TIFF image. "
                    f"The selected file has shape {data.shape}."
                )

            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("The TIFF must contain numeric data.")

            if np.iscomplexobj(data):
                raise ValueError("Complex-valued TIFF data is unsupported.")

            finite = data[np.isfinite(data)]
            if finite.size == 0:
                raise ValueError("The image contains no finite intensities.")

            # Initial contrast similar to mask_widget.py.
            low = float(np.min(finite))
            high = float(np.percentile(finite, 99.9))
            if high <= low:
                high = low + 1.0

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Unable to Load TIFF", str(exc)
            )
            return

        self.image_data = data
        self.image_file_name = filename
        # A new image must be calibrated independently.
        self.calibrated_geometry = None
        self.btn_pick_center.setChecked(False)

        height, width = data.shape

        self.tracked_points_x.clear()
        self.tracked_points_y.clear()
        self.tracked_rings.clear()
        self.spoke_pick_scatter.setData(x=[], y=[])
        self.update_ring_summary()

        # A neutral initial center estimate, not a calibration result.
        self.inputs["Beam Center X (px) "][0].setText(
            f"{(width - 1) / 2:.4f}"
        )
        self.inputs["Beam Center Y (px) "][0].setText(
            f"{(height - 1) / 2:.4f}"
        )

        radius = min(height, width) / 2
        self.txt_inner_edge.setText(f"{0.02 * radius:.1f}")
        self.txt_outer_edge.setText(f"{0.9 * radius:.1f}")

        self.image.setImage(
            data,
            autoLevels=False,
            levels=(low, high),
        )
        self.intensity_bar.setLevels(low, high)

        self.view.setTitle(os.path.basename(filename))
        self.view.setRange(
            xRange=(0, width),
            yRange=(0, height),
            padding=0.02,
        )

        self.lbl_file.setText(
            f"{width} × {height} pixels"
        )

        for button in (
            self.btn_pick_center,
            self.btn_spoke_calibrate,
            self.btn_refine,
            self.btn_reoptimize,
            self.btn_clear,
            self.btn_export_imctrl,
            self.btn_save_pipeline,
        ):
            button.setEnabled(True)
        self.btn_export_imctrl.setEnabled(False)
        self.btn_save_pipeline.setEnabled(False)

        self.coord_label.setText("")
        self.lbl_status.setText(
            "Status: TIFF loaded. Select the correct calibrant for the image, then click near the ring-to-trace to trace it. Doublecheck the ring-to-trace selection on the right panel."
        )
        self.plot_image()

    # ==================================================================
    # OVERLAYS AND COORDINATE READOUT
    # ==================================================================

    def plot_image(self, *_):
        # Keep the calibrated beam-center marker synchronized with the image.
        self.update_beam_center_marker()
        # Update overlays only. Do not reset contrast or zoom.
        self.tracked_scatter.setData(
            x=np.asarray(self.tracked_points_x),
            y=np.asarray(self.tracked_points_y),
        )

        self.inner_curve.hide()
        self.outer_curve.hide()
        for item in self.integration_limit_items:
            item.hide()

        if self.image_data is None:
            return

        self.update_integration_limit_overlay()

        if self.chk_show_edges.isChecked():
            try:
                x0 = float(
                    self.inputs["Beam Center X (px) "][0].text()
                )
                y0 = float(
                    self.inputs["Beam Center Y (px) "][0].text()
                )
                r_in = float(self.txt_inner_edge.text())
                r_out = float(self.txt_outer_edge.text())

                theta = np.linspace(0, 2 * np.pi, 361)

                self.inner_curve.setData(
                    x0 + r_in * np.cos(theta),
                    y0 + r_in * np.sin(theta),
                )
                self.outer_curve.setData(
                    x0 + r_out * np.cos(theta),
                    y0 + r_out * np.sin(theta),
                )

                self.inner_curve.show()
                self.outer_curve.show()

            except ValueError:
                pass

    def update_integration_limit_overlay(self):
        if not self.chk_show_integration_limits.isChecked():
            return

        try:
            center_x, center_y, distance, tilt, rotation, penetration = (
                self.current_geometry()
            )
            pixel_size_x = float(self.txt_pixel_x.text())
            pixel_size_y = float(self.txt_pixel_y.text())
            inner_angle = np.radians(self.spin_inner_tth.value())
            outer_angle = np.radians(self.spin_outer_tth.value())
            azimuth_start = self.spin_azimuth_start.value()
            azimuth_end = self.spin_azimuth_end.value()

            if inner_angle >= outer_angle:
                return
            if not (0 <= center_x < self.image_data.shape[1]
                    and 0 <= center_y < self.image_data.shape[0]):
                return

            raw_span = azimuth_end - azimuth_start
            if abs(raw_span) >= 360.0 - 1e-9:
                azimuth_span = 360.0
            elif raw_span > 0:
                azimuth_span = raw_span
            elif raw_span < 0:
                azimuth_span = raw_span % 360.0
            else:
                return

            phi_values = np.linspace(
                0.0, 2.0 * np.pi, 721, endpoint=True
            )
            phi_degrees = np.degrees(phi_values)
            tilt_radians = np.radians(tilt)
            rotation_radians = np.radians(rotation - 90.0)
            cos_rotation = np.cos(rotation_radians)
            sin_rotation = np.sin(rotation_radians)

            # The azimuth transform is linear in detector-plane offsets, so
            # its result is independent of radius along each detector ray.
            dx = pixel_size_x * np.cos(phi_values)
            dy = pixel_size_y * np.sin(phi_values)
            rotated_x = dx * cos_rotation + dy * sin_rotation
            rotated_y = -dx * sin_rotation + dy * cos_rotation
            transverse_x = rotated_x * np.cos(tilt_radians)
            lab_x = transverse_x * cos_rotation - rotated_y * sin_rotation
            lab_y = transverse_x * sin_rotation + rotated_y * cos_rotation
            azimuth_values = (
                np.degrees(np.arctan2(lab_y, lab_x)) % 360.0
            )

            def nearest_azimuth_index(target):
                difference = (
                    (azimuth_values - target + 180.0) % 360.0
                ) - 180.0
                return int(np.argmin(np.abs(difference)))

            start_phi_index = nearest_azimuth_index(azimuth_start)
            end_phi_index = nearest_azimuth_index(azimuth_end)

            if azimuth_span >= 360.0 - 1e-9:
                in_sector = np.ones(phi_values.size, dtype=bool)
            else:
                azimuth_stop = (azimuth_start + azimuth_span) % 360.0
                if azimuth_start <= azimuth_stop:
                    in_sector = (
                        (azimuth_values >= azimuth_start)
                        & (azimuth_values <= azimuth_stop)
                    )
                else:
                    in_sector = (
                        (azimuth_values >= azimuth_start)
                        | (azimuth_values <= azimuth_stop)
                    )
            in_sector[start_phi_index] = True
            in_sector[end_phi_index] = True

            height, width = self.image_data.shape
            inner_radii = np.full(phi_values.size, np.nan)
            outer_radii = np.full(phi_values.size, np.nan)
            edge_radii = np.full(phi_values.size, np.nan)

            for index, (phi, visible) in enumerate(
                zip(phi_values, in_sector)
            ):
                if not visible:
                    continue

                unit_x, unit_y = np.cos(phi), np.sin(phi)
                limits = []
                if unit_x > 1e-12:
                    limits.append((width - 1 - center_x) / unit_x)
                elif unit_x < -1e-12:
                    limits.append(-center_x / unit_x)
                if unit_y > 1e-12:
                    limits.append((height - 1 - center_y) / unit_y)
                elif unit_y < -1e-12:
                    limits.append(-center_y / unit_y)
                max_radius = min(limits)
                if max_radius <= 0:
                    continue
                edge_radii[index] = max_radius

                def angle_error(radius, target):
                    return float(gsas2_metric_projection(
                        np.array([center_x + radius * unit_x]),
                        np.array([center_y + radius * unit_y]),
                        center_x * pixel_size_x,
                        center_y * pixel_size_y,
                        distance,
                        tilt,
                        rotation,
                        pixel_size_x,
                        pixel_size_y,
                        penetration=penetration,
                    )[0] - target)

                for target, result_radii in (
                    (inner_angle, inner_radii),
                    (outer_angle, outer_radii),
                ):
                    try:
                        if angle_error(max_radius, target) < 0:
                            continue
                        result_radii[index] = brentq(
                            lambda radius: angle_error(radius, target),
                            0.0,
                            max_radius,
                        )
                    except (ValueError, FloatingPointError):
                        continue

            for item, radii in zip(
                self.integration_limit_items[:2],
                (inner_radii, outer_radii),
            ):
                item.setData(
                    center_x + radii * np.cos(phi_values),
                    center_y + radii * np.sin(phi_values),
                )
                if np.any(np.isfinite(radii)):
                    item.show()

            for item, phi_index in zip(
                self.integration_limit_items[2:],
                (start_phi_index, end_phi_index),
            ):
                inner_radius = inner_radii[phi_index]
                outer_radius = outer_radii[phi_index]
                edge_radius = edge_radii[phi_index]

                if not np.isfinite(inner_radius):
                    inner_radius = 0.0
                if not np.isfinite(outer_radius):
                    outer_radius = edge_radius

                if (
                    not np.isfinite(outer_radius)
                    or outer_radius <= inner_radius
                ):
                    continue
                phi = phi_values[phi_index]
                radii = np.asarray([inner_radius, outer_radius])
                item.setData(
                    center_x + radii * np.cos(phi),
                    center_y + radii * np.sin(phi),
                )
                item.show()

        except (ValueError, FloatingPointError):
            for item in self.integration_limit_items:
                item.hide()

    def mouse_moved(self, scene_pos):
        """Show pixel intensity and calibrated scattering coordinates."""
        if self.image_data is None:
            self.coord_label.setText("")
            return
    
        # Restrict the readout to the actual image ViewBox.
        if not self.view.vb.sceneBoundingRect().contains(scene_pos):
            self.coord_label.setText("")
            return
    
        point = self.view.vb.mapSceneToView(scene_pos)
        display_x = float(point.x())
        display_y = float(point.y())
    
        # ImageItem pixels occupy [x, x+1) and [y, y+1).
        pixel_x = int(np.floor(display_x))
        pixel_y = int(np.floor(display_y))
    
        height, width = self.image_data.shape
        if not (0 <= pixel_x < width and 0 <= pixel_y < height):
            self.coord_label.setText("")
            return
    
        intensity = self.image_data[pixel_y, pixel_x]
        readout = (
            f"x={pixel_x}, y={pixel_y}, "
            f"Intensity={intensity:.4g}"
        )
    
        if self.calibrated_geometry is not None:
            # Evaluate geometry at the continuous mouse position, not the
            # rounded pixel index. This matches the ring-picking coordinates.
            two_theta, azimuth, d_spacing, q_value = (
                self.scattering_coordinates_at(display_x, display_y)
            )
    
            def formatted(value, decimals):
                """Format undefined/infinite quantities for the UI."""
                if np.isnan(value):
                    return "—"
                if np.isinf(value):
                    return "∞" if valueR > 0 else "−∞"
                return f"{value:.{decimals}f}"
    
            readout += (
                f"   |   2θ={formatted(two_theta, 4)}°"
                f"   azm={formatted(azimuth, 2)}°"
                f"   d={formatted(d_spacing, 4)} Å"
                f"   Q={formatted(q_value, 4)} Å⁻¹"
            )
    
        self.coord_label.setText(readout)

    # ==================================================================
    # SINGLE-CLICK RING TRACKING
    # ==================================================================

    def handle_mouse_click_pick(self, event):
        if self.image_data is None:
            return

        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        if not self.view.sceneBoundingRect().contains(
            event.scenePos()
        ):
            return

        point = self.view.vb.mapSceneToView(event.scenePos())
        height, width = self.image_data.shape

        if not (0 <= point.x() < width and 0 <= point.y() < height):
            return

        if self.pick_beam_center_mode:
            center_x = float(point.x())
            center_y = float(point.y())
            self.inputs["Beam Center X (px) "][0].setText(
                f"{center_x:.4f}"
            )
            self.inputs["Beam Center Y (px) "][0].setText(
                f"{center_y:.4f}"
            )
            self.calibrated_geometry = None
            self.btn_export_imctrl.setEnabled(False)
            self.btn_save_pipeline.setEnabled(False)
            self.btn_pick_center.setChecked(False)
            self.plot_image()
            self.lbl_status.setText(
                f"Status: Estimated beam center set to "
                f"({center_x:.2f}, {center_y:.2f}) px."
            )
            return

        try:
            x0 = float(
                self.inputs["Beam Center X (px) "][0].text()
            )
            y0 = float(
                self.inputs["Beam Center Y (px) "][0].text()
            )
        except ValueError:
            self.set_error_status(
                "Error: Beam center coordinates must be numeric."
            )
            return

        self.lbl_status.setText("Status: Tracking ring...")
        QtWidgets.QApplication.processEvents()

        r_seed = np.hypot(point.x() - x0, point.y() - y0)
        start_angle = np.arctan2(point.y() - y0, point.x() - x0)

        self.trace_ring_from_radius_seed(
            x0, y0, r_seed, start_angle=start_angle
        )

    def toggle_beam_center_pick(self, enabled):
        self.pick_beam_center_mode = enabled
        if enabled:
            self.lbl_status.setText(
                "Status: Click the estimated beam center in the image."
            )
        else:
            self.lbl_status.setText(
                "Status: Beam-center picking cancelled."
            )

    def trace_ring_from_radius_seed(
        self, x0, y0, r_seed, start_angle=0.0
    ):
        """Track one radial intensity peak with continuity constraints."""
        if self.image_data is None:
            return

        height, width = self.image_data.shape

        # Convert once: interpolation should operate on floating-point data.
        data = np.asarray(self.image_data, dtype=float)

        # Settings exposed in the panel.
        n_angles = self.spin_trace_angles.value()
        radial_step = self.spin_radial_step.value()
        search_half_width = self.spin_search_width.value()

        # The initial pick may select any peak within the search window.
        initial_tolerance = search_half_width

        # Once tracking begins, retain a tighter continuity constraint.
        max_prediction_error = min(1.5, search_half_width)

        # These continuity settings remain fixed for now.
        max_slope = 0.75
        max_missing_angles = 60

        angles = start_angle + np.linspace(
            0.0, 2.0 * np.pi, n_angles, endpoint=False
        )

        last_radius = float(r_seed)
        radial_slope = 0.0
        missing = 0
        started = False

        new_x = []
        new_y = []

        for phi in angles:
            # Predict the next radius from the recent local trend.
            # Limit extrapolation across gaps.
            predicted_radius = last_radius + radial_slope * min(
                missing + 1, 4
            )

            radii = np.arange(
                max(0.0, predicted_radius - search_half_width),
                predicted_radius + search_half_width + radial_step,
                radial_step,
            )

            sample_x = x0 + radii * np.cos(phi)
            sample_y = y0 + radii * np.sin(phi)

            valid = (
                (sample_x >= 0)
                & (sample_x <= width - 1)
                & (sample_y >= 0)
                & (sample_y <= height - 1)
            )

            radii = radii[valid]
            sample_x = sample_x[valid]
            sample_y = sample_y[valid]

            if radii.size < 5:
                missing += 1
                if missing > max_missing_angles:
                    break
                continue

            profile = self.sample_display_profile(
                data,
                sample_x,
                sample_y,
            )

            if not np.all(np.isfinite(profile)):
                missing += 1
                if missing > max_missing_angles:
                    break
                continue

            # Use the panel's smoothing and peak-detection settings.
            smooth, peaks = self.detect_radial_peaks(
                profile, radial_step
            )

            candidate_radii = []
            tolerance = (
                initial_tolerance
                if not started
                else max_prediction_error
            )

            for peak in peaks:
                # Parabolic sub-sample peak refinement.
                left = smooth[peak - 1]
                middle = smooth[peak]
                right = smooth[peak + 1]

                denominator = left - 2.0 * middle + right
                offset = 0.0

                if denominator != 0:
                    offset = 0.5 * (left - right) / denominator
                    offset = float(np.clip(offset, -0.5, 0.5))

                radius = radii[peak] + offset * radial_step

                if abs(radius - predicted_radius) <= tolerance:
                    candidate_radii.append(radius)

            if not candidate_radii:
                # No acceptable continuation: do not jump to another ring.
                missing += 1
                if missing > max_missing_angles:
                    break
                continue

            selected_radius = min(
                candidate_radii,
                key=lambda radius: abs(radius - predicted_radius),
            )

            if started:
                measured_slope = (
                    selected_radius - last_radius
                ) / (missing + 1)

                radial_slope = float(
                    np.clip(
                        0.5 * radial_slope + 0.5 * measured_slope,
                        -max_slope,
                        max_slope,
                    )
                )

            last_radius = selected_radius
            missing = 0
            started = True

            new_x.append(x0 + selected_radius * np.cos(phi))
            new_y.append(y0 + selected_radius * np.sin(phi))

        # Replace the previous selection, avoiding accidental mixing of rings.
        self.store_traced_ring(new_x, new_y)

        if new_x:
            self.lbl_status.setText(
                f"Status: Tracked {len(new_x)} points on one ring. "
                "Check and confirm the dots are all on the target ring. Then click Calibrate."
            )
        else:
            self.lbl_status.setText(
                "Status: No suitable peak found near the click. "
                "Click directly on a bright part of the ring."
            )

        self.plot_image()

    def clear_tracked_nodes(self):
        self.tracked_rings.clear()
        self.tracked_points_x = []
        self.tracked_points_y = []
        self.spoke_pick_scatter.setData(x=[], y=[])

        self.update_ring_summary()
        self.lbl_status.setText("Status: Cleared all tracked rings.")
        self.plot_image()

    def export_calibration_to_imctrl(self):
        """Write the current calibrated geometry in GSAS-II imctrl format."""
        calibration = self.calibrated_geometry
        if calibration is None:
            self.set_error_status(
                "Error: Calibrate the detector before exporting parameters."
            )
            return

        calibrant_names = {
            "Silicon (SRM 640c)": "Si  SRM640c",
            "CeO2 (NIST SRM 674b)": "CeO2  SRM674b",
            "LaB6 (NIST SRM 660a)": "LaB6  SRM660a",
            "Silver Behenate": "Ag Behenate",
        }
        calibrant = self.combo_calibrant.currentText()
        if calibrant not in calibrant_names:
            self.set_error_status(
                "Error: No GSAS-II calibrant label is mapped for this standard."
            )
            return

        parameters = calibration["parameters"]
        center_x = float(parameters[0]) * calibration["pixel_size_x"]
        center_y = float(parameters[1]) * calibration["pixel_size_y"]
        distance = float(parameters[2])
        tilt = float(parameters[3])
        rotation = float(parameters[4])
        penetration = float(parameters[5])
        wavelength = float(calibration["wavelength"])
        inner_tth = self.spin_inner_tth.value()
        outer_tth = self.spin_outer_tth.value()
        azimuth_start = self.spin_azimuth_start.value()
        azimuth_end = self.spin_azimuth_end.value()
        integration_bins = self.spin_integration_bins.value()
        output_azimuth_end = azimuth_end
        if output_azimuth_end < azimuth_start:
            output_azimuth_end += 360.0

        if inner_tth >= outer_tth:
            self.set_error_status(
                "Error: Inner 2-theta must be less than outer 2-theta."
            )
            return

        if integration_bins < 1:
            self.set_error_status(
                "Error: Number of integration bins must be positive."
            )
            return

        full_azimuth = (
            output_azimuth_end - azimuth_start >= 360.0 - 1e-9
        )

        vary = {
            "dist": self.inputs["Distance (mm) "][1].isChecked(),
            "det-X": self.inputs["Beam Center X (px) "][1].isChecked(),
            "det-Y": self.inputs["Beam Center Y (px) "][1].isChecked(),
            "tilt": self.inputs["Tilt (deg) "][1].isChecked(),
            "phi": self.inputs["Tilt Rotation (deg) "][1].isChecked(),
            "dep": self.inputs["Penetration (fraction) "][1].isChecked(),
            "wave": self.inputs[WAVELENGTH_KEY][1].isChecked(),
        }
        vary_list = "{" + ", ".join(
            f"'{key}': {value}" for key, value in vary.items()
        ) + "}"

        image_path = self.image_file_name or "calibration.tif"
        directory = os.path.dirname(image_path)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        default_path = os.path.join(directory, f"{image_stem}.imctrl")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export GSAS-II Image Controls",
            default_path,
            "GSAS-II image controls (*.imctrl)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".imctrl"):
            filename += ".imctrl"

        lines = [
            "type:PWDR",
            "color:gist_earth",
            f"wavelength:{wavelength:.12g}",
            f"calibrant:{calibrant_names[calibrant]}",
            f"distance:{distance:.12g}",
            f"center:[{center_x:.12g}, {center_y:.12g}]",
            "Oblique:[0.5, False]",
            f"tilt:{tilt:.12g}",
            f"rotation:{rotation:.12g}",
            "azmthOff:0.0",
            f"fullIntegrate:{full_azimuth}",
            f"LRazimuth:[{azimuth_start:.12g}, {output_azimuth_end:.12g}]",
            f"setdist:{distance:.12g}",
            f"IOtth:[{inner_tth:.12g}, {outer_tth:.12g}]",
            f"outChannels:{integration_bins}",
            "outAzimuths:1",
            "invert_x:False",
            "invert_y:True",
            f"DetDepth:{penetration:.12g}",
            "calibskip:0",
            "pixLimit:2",
            "cutoff:1.0",
            "calibdmin:1.0",
            "Flat Bkg:0.0",
            f"varyList:{vary_list}",
            "binType:2-theta",
            "SampleShape:Cylinder",
            "PolaVal:[0.99, False]",
            "SampleAbs:[0.0, False]",
            "dark image:['', -1.0]",
            "background image:['', -1.0]",
        ]

        try:
            with open(filename, "w", encoding="utf-8", newline="\n") as output:
                output.write("\n".join(lines) + "\n")
        except OSError as exc:
            self.set_error_status(f"Error: Could not write imctrl file: {exc}")
            return

        self.lbl_status.setText(
            f"Status: Calibration parameters exported to {os.path.basename(filename)}."
        )
        return filename  # Return the filename after export

    def save_and_export_to_pipeline(self):
        """Export calibrated controls, return the path to the pipeline, and close."""
        if not self.pipeline_handoff_file:
            QtWidgets.QMessageBox.warning(
                self,
                "Pipeline Not Connected",
                "This action is available when opened from the pipeline.",
            )
            return

        filename = self.export_calibration_to_imctrl()
        if not filename:
            return

        try:
            with open(
                self.pipeline_handoff_file,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handoff:
                handoff.write(os.path.abspath(filename))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Unable to Return Config File",
                str(exc),
            )
            return

        self.lbl_status.setText(
            "Status: Calibration saved. Returning to the pipeline..."
        )
        self.close()

    # ==========================================================================
    # NONLINEAR LEAST-SQUARES REFINEMENT
    # ==========================================================================

    def _refine_collected_rings(self):
        """Jointly fit one detector geometry to all assigned rings."""
        if not self.tracked_rings:
            self.set_error_status(
                "Error: Trace and assign rings before refining."
            )
            return

        param_keys = FIT_KEYS

        try:
            pixel_size_x = float(self.txt_pixel_x.text())
            pixel_size_y = float(self.txt_pixel_y.text())
            wavelength = float(self.txt_wave.text())
            lattice = float(self.txt_lattice.text())

            if not np.all(np.isfinite(
                [pixel_size_x, pixel_size_y, wavelength, lattice]
            )):
                raise ValueError("Hardware inputs must be finite.")

            if min(pixel_size_x, pixel_size_y, wavelength, lattice) <= 0:
                raise ValueError(
                    "Pixel sizes, wavelength, and lattice parameter "
                    "must be positive."
                )

            p0 = np.array([
                float(self.inputs[key][0].text())
                for key in param_keys
            ])

            if not np.all(np.isfinite(p0)):
                raise ValueError("Geometry inputs must be finite.")

            refine_mask = np.array([
                self.inputs[key][1].isChecked()
                for key in param_keys
            ], dtype=bool)

            if not np.any(refine_mask):
                raise ValueError(
                    "Select at least one parameter for refinement."
                )

            cal_name = self.combo_calibrant.currentText()
            system = CALIBRANTS[cal_name]["system"]

            all_x = []
            all_y = []
            # Target scattering angles must be recomputed as wavelength changes.
            all_d_spacings = []

            for index, ring in enumerate(
                self.tracked_rings, start=1
            ):
                if ring["calibrant"] != cal_name:
                    raise ValueError(
                        "Collected rings belong to another calibrant. "
                        "Clear the points and pick rings for the "
                        "currently selected standard."
                    )

                h, k, l = ring["hkl"]

                if system == "Cubic":
                    d_spacing = lattice / np.sqrt(
                        h * h + k * k + l * l
                    )
                else:
                    # Silver Behenate layer-spacing approximation.
                    if h != 0 or k != 0 or l <= 0:
                        raise ValueError(
                            "The layered model supports only (00l), l > 0."
                        )
                    d_spacing = lattice / l

                bragg_argument = wavelength / (2.0 * d_spacing)

                if not 0 < bragg_argument <= 1:
                    raise ValueError(
                        f"Ring {index}, reflection {ring['hkl']}: "
                        "inaccessible at the entered wavelength."
                    )

                target = 2.0 * np.arcsin(bragg_argument)

                x = ring["x"]
                y = ring["y"]

                all_x.append(x)
                all_y.append(y)
                all_d_spacings.append(
                    np.full(x.size, d_spacing, dtype=float)
                )

            pts_x = np.concatenate(all_x)
            pts_y = np.concatenate(all_y)
            point_d_spacings = np.concatenate(all_d_spacings)

            if pts_x.size < np.count_nonzero(refine_mask):
                raise ValueError(
                    "Too few tracked points for the selected parameters."
                )

            # Bounds avoid negative distance and extreme detector tilt.
            # The application currently fits forward-scattering rings only.
            # Therefore wavelength must remain below sqrt(2) * min(d).
            # A small margin avoids the exact 90-degree boundary.
            maximum_wavelength = (
                np.sqrt(2.0)
                * float(np.min(point_d_spacings))
                * (1.0 - 1e-10)
            )
            
            lower = np.array([
                -np.inf,   # Center X
                -np.inf,   # Center Y
                1e-6,      # Distance, mm
                -85.0,     # Tilt, degrees
                -np.inf,   # Tilt rotation, degrees
                0.0,       # Penetration, dimensionless
                1e-12,     # Wavelength, Å
            ])
            
            upper = np.array([
                np.inf,
                np.inf,
                np.inf,
                85.0,
                np.inf,
                1.0,       # Application penetration guardrail
                maximum_wavelength,
            ])

            # Validate fixed parameters as well as refined parameters.
            if np.any(p0 < lower) or np.any(p0 > upper):
                raise ValueError(
                    "Initial geometry is outside the fit bounds: "
                    "Require distance > 0, tilt between -85 and 85 degrees, "
                    "penetration between 0 and 1, and a positive wavelength "
                    "that keeps all collected rings below 2θ = 90°."
                )

            initial_guess = p0[refine_mask]
            
            def residuals(active_params):
                """Compare geometric angles with wavelength-dependent Bragg angles."""
                current = p0.copy()
                current[refine_mask] = active_params
            
                (
                    cx,
                    cy,
                    distance,
                    tilt,
                    rotation,
                    penetration,
                    current_wavelength,
                ) = current
            
                calculated = gsas2_metric_projection(
                    pts_x,
                    pts_y,
                    cx * pixel_size_x,
                    cy * pixel_size_y,
                    distance,
                    tilt,
                    rotation,
                    pixel_size_x,
                    pixel_size_y,
                    penetration=penetration,
                )
            
                # Recompute the reference angles on every optimizer evaluation.
                target_angles = 2.0 * np.arcsin(
                    current_wavelength / (2.0 * point_d_spacings)
                )

                return calculated - target_angles

            self.lbl_status.setText(
                f"Status: Jointly refining "
                f"{len(self.tracked_rings)} rings, "
                f"{pts_x.size} points..."
            )
            QtWidgets.QApplication.processEvents()

            result = least_squares(
                residuals,
                initial_guess,
                method="trf",
                bounds=(
                    lower[refine_mask],
                    upper[refine_mask],
                ),
                x_scale="jac",
                max_nfev=3000,
            )

            if not result.success:
                self.lbl_status.setText(
                    f"Status: Refinement did not converge: "
                    f"{result.message}"
                )
                return

            final = p0.copy()
            final[refine_mask] = result.x
            
            # Rotation is periodic; report it in the same 0–360° range.
            final[4] %= 360.0
            
            self.pixel_size_x = pixel_size_x
            self.pixel_size_y = pixel_size_y
            fitted_wavelength = float(final[-1])
            
            for index, key in enumerate(param_keys):
                self.inputs[key][0].setText(f"{final[index]:.12g}")
            
            # Keep geometry at six entries so the marker and mouse readout
            # continue to interpret penetration correctly.
            self.calibrated_geometry = {
                "parameters": final[:len(GEOMETRY_KEYS)].copy(),
                "pixel_size_x": float(pixel_size_x),
                "pixel_size_y": float(pixel_size_y),
                "wavelength": fitted_wavelength,
            }
            self.btn_export_imctrl.setEnabled(True)
            self.btn_save_pipeline.setEnabled(
                self.pipeline_handoff_file is not None
            )

            self.update_ring_summary()
            
            angular_errors = residuals(result.x)
            rms_deg = np.degrees(
                np.sqrt(np.mean(angular_errors ** 2))
            )
            
            self.lbl_status.setText(
                f"Status: Joint fit converged using "
                f"{len(self.tracked_rings)} rings and {pts_x.size} points. "
                f"Angular RMS: {rms_deg:.5g}°. "
                f"λ={fitted_wavelength:.8g} Å."
            )
            
            self.plot_image()
            return True

        except (ValueError, FloatingPointError) as exc:
            self.set_error_status(f"Error: {exc}")

    # ==========================================================================
    # Generate expected reflections
    # This extends beyond the eight entries in your original dictionary. Equivalent cubic reflections with the same
    # d-spacing are grouped into one powder ring. These rules generate candidate positions, not intensities. 
    # A predicted reflection can still be too weak to detect.
    # ==========================================================================

    def expansion_reflections(self, minimum_d, lattice):
        """Return expected powder-ring HKLs and d-spacings."""
        name = self.combo_calibrant.currentText()
        system = CALIBRANTS[name]["system"]

        if system != "Cubic":
            # Silver Behenate: layer harmonics (00l).
            count = int(np.floor(lattice / minimum_d))
            if count > 500:
                raise ValueError(
                    "Minimum d-spacing generates too many layer harmonics."
                )

            return [
                ((0, 0, l), lattice / l)
                for l in range(1, count + 1)
            ]

        maximum_index = int(np.floor(lattice / minimum_d))

        if maximum_index > 60:
            raise ValueError(
                "Minimum d-spacing is too small for this reflection search."
            )

        rings = {}

        for h in range(maximum_index + 1):
            for k in range(h + 1):
                for l in range(k + 1):
                    n = h * h + k * k + l * l

                    if n == 0:
                        continue

                    d = lattice / np.sqrt(n)
                    if d < minimum_d:
                        continue

                    same_parity = (h % 2 == k % 2 == l % 2)

                    if name.startswith("Silicon"):
                        # Diamond: FCC parity plus the even-index extinction.
                        if not same_parity:
                            continue
                        if h % 2 == 0 and (h + k + l) % 4 != 0:
                            continue

                    elif name.startswith(("CeO2", "Ni")):
                        # Fluoride: FCC translational extinctions.
                        if not same_parity:
                            continue

                    elif not name.startswith("LaB6"):
                        raise ValueError(
                            "No reflection-generation rules for this calibrant."
                        )

                    # LaB6 is primitive cubic. Actual intensities are not
                    # predicted here; weak/absent rings are rejected later.
                    rings.setdefault(n, (h, k, l))

        return [
            (rings[n], lattice / np.sqrt(n))
            for n in sorted(rings)
        ]

    # Read the current geometry

    def current_geometry(self):
        keys = GEOMETRY_KEYS

        values = np.array([
            float(self.inputs[key][0].text())
            for key in keys
        ])

        if not np.all(np.isfinite(values)):
            raise ValueError("Geometry parameters must be finite.")
            
        if values[2] <= 0:
            raise ValueError("Distance must be positive.")

        if not -85.0 <= values[3] <= 85.0:
            raise ValueError("Tilt must be between -85 and 85 degrees.")

        if not 0.0 <= values[5] <= 1.0:
            raise ValueError(
                "Penetration must be between 0 and 1."
            )

        return values


    # Predict and trace one ring
    # At each azimuth, this solves for the expected radius, then searches near that radius. It does not carry the last observed radius
    # forward around the detector. The acceptance thresholds are conservative defaults. 
    # Spotty diffraction or strongly masked images may require a different acceptance criterion.

    def trace_predicted_ring(self, target_angle, search_width=None):
        """Find peaks close to one geometry-predicted diffraction ring."""
        cx, cy, distance, tilt, rotation, penetration = self.current_geometry()
        pixel_size_x = float(self.txt_pixel_x.text())
        pixel_size_y = float(self.txt_pixel_y.text())

        data = np.asarray(self.image_data, dtype=float)
        height, width = data.shape

        if not (0 <= cx <= width - 1 and 0 <= cy <= height - 1):
            raise ValueError(
                "Expansion currently requires the beam center inside the image."
            )

        points_x = []
        points_y = []
        visible_directions = 0

        if search_width is None:
            search_width = self.spin_search_width.value()

        radial_step = self.spin_radial_step.value()
        n_angles = self.spin_trace_angles.value()

        for phi in np.linspace(
            0, 2 * np.pi, n_angles, endpoint=False
        ):
            ux, uy = np.cos(phi), np.sin(phi)

            # Distance along this ray to the image boundary.
            limits = []

            if ux > 1e-12:
                limits.append((width - 1 - cx) / ux)
            elif ux < -1e-12:
                limits.append(-cx / ux)

            if uy > 1e-12:
                limits.append((height - 1 - cy) / uy)
            elif uy < -1e-12:
                limits.append(-cy / uy)

            maximum_radius = min(limits)

            if maximum_radius <= 0:
                continue

            # Reject rays crossing the forward model's denominator
            # singularity; its arctan implementation is not valid there.
            # This boundary check must use the same internal direction as
            # gsas2_metric_projection(), not the rotation-axis direction directly.
            
            tilt_direction_rad = np.radians(rotation -90.0)
            
            plane_offset = (
                maximum_radius
                * (
                    pixel_size_x * ux * np.cos(tilt_direction_rad)
                    + pixel_size_y * uy * np.sin(tilt_direction_rad)
                )
            )
            denominator_end = (
                distance - plane_offset * np.sin(np.radians(tilt))
            )

            if distance <= 0 or denominator_end <= 0:
                continue

            def angular_error(radius):
                angle = gsas2_metric_projection(
                    np.array([cx + radius * ux]),
                    np.array([cy + radius * uy]),
                    cx * pixel_size_x,
                    cy * pixel_size_y,
                    distance,
                    tilt,
                    rotation,
                    pixel_size_x,
                    pixel_size_y,
                    penetration=penetration,
                )[0]
                return angle - target_angle

            if angular_error(maximum_radius) < 0:
                continue

            predicted_radius = brentq(
                angular_error, 0.0, maximum_radius
            )

            # Require a complete local search profile.
            if (
                predicted_radius <= search_width
                or predicted_radius + search_width >= maximum_radius
            ):
                continue

            visible_directions += 1

            radii = np.arange(
                predicted_radius - search_width,
                predicted_radius + search_width + radial_step,
                radial_step,
            )

            # Convert display coordinates to array interpolation
            # coordinates, including the optional half-pixel correction.
            profile = self.sample_display_profile(
                data,
                cx + radii * ux,
                cy + radii * uy,
            )

            if not np.all(np.isfinite(profile)):
                continue

            # Read smoothing, prominence, noise threshold, and peak
            # separation from the panel.
            smooth, peaks = self.detect_radial_peaks(
                profile, radial_step
            )
            
            if peaks.size == 0:
                continue

            peak = min(
                peaks,
                key=lambda i: abs(radii[i] - predicted_radius),
            )

            radius = radii[peak]

            # Subpixel parabolic peak correction.
            left, middle, right = smooth[peak - 1:peak + 2]
            denominator = left - 2 * middle + right

            if denominator != 0:
                offset = np.clip(
                    0.5 * (left - right) / denominator,
                    -0.5,
                    0.5,
                )
                radius += offset * radial_step

            points_x.append(cx + radius * ux)
            points_y.append(cy + radius * uy)

        # Require support across multiple azimuths, not isolated noise.
        required = max(30, int(0.35 * visible_directions))

        if len(points_x) < required:
            return None

        return (
            np.asarray(points_x),
            np.asarray(points_y),
        )

    def fit_spoke_ellipse(self, points_x, points_y):
        """Fit an ellipse by five-point consensus and reject outliers."""
        points_x = np.asarray(points_x, dtype=float)
        points_y = np.asarray(points_y, dtype=float)
        points = np.column_stack((points_x, points_y))

        if points.shape[0] < 5:
            raise ValueError(
                "At least five spoke profiles must find a ring peak."
            )

        maximum_axis = 2.0 * max(self.image_data.shape)
        bounds = (
            [-np.inf, -np.inf, np.log(0.5), np.log(0.5), -np.inf],
            [np.inf, np.inf, np.log(maximum_axis),
             np.log(maximum_axis), np.inf],
        )

        def ellipse_residual(parameters, selected_points):
            cx, cy, log_major, log_minor, angle = parameters
            major = np.exp(log_major)
            minor = np.exp(log_minor)
            cosine, sine = np.cos(angle), np.sin(angle)
            dx = selected_points[:, 0] - cx
            dy = selected_points[:, 1] - cy
            local_major = cosine * dx + sine * dy
            local_minor = -sine * dx + cosine * dy
            normalized_radius = np.sqrt(
                (local_major / major) ** 2
                + (local_minor / minor) ** 2
            )
            return (normalized_radius - 1.0) * np.sqrt(major * minor)

        def initial_guess(selected_points):
            origin = np.mean(selected_points, axis=0)
            scale = max(float(np.max(np.ptp(selected_points, axis=0))), 1.0)
            normalized = (selected_points - origin) / scale
            x = normalized[:, 0]
            y = normalized[:, 1]
            design = np.column_stack((x * x, x * y, y * y, x, y))
            coefficients = np.linalg.solve(
                design, np.ones(selected_points.shape[0])
            )
            quadratic = np.array([
                [coefficients[0], 0.5 * coefficients[1]],
                [0.5 * coefficients[1], coefficients[2]],
            ])
            linear = coefficients[3:5]
            eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
            rhs = 1.0

            if np.all(eigenvalues < 0):
                quadratic = -quadratic
                linear = -linear
                eigenvalues = -eigenvalues[::-1]
                eigenvectors = eigenvectors[:, ::-1]
                rhs = -1.0
            elif not np.all(eigenvalues > 0):
                raise ValueError("Five points do not define an ellipse.")

            center_normalized = -0.5 * np.linalg.solve(
                quadratic, linear
            )
            scale_squared = rhs + center_normalized @ quadratic @ center_normalized
            if not np.isfinite(scale_squared) or scale_squared <= 0:
                raise ValueError("Five points do not define a real ellipse.")

            axes = np.sqrt(scale_squared / eigenvalues) * scale
            center = origin + center_normalized * scale
            major = float(axes[0])
            minor = float(axes[1])
            angle = np.arctan2(
                eigenvectors[1, 0], eigenvectors[0, 0]
            )

            if max(major, minor) > maximum_axis:
                raise ValueError("Ellipse axes exceed image bounds.")

            return [
                center[0],
                center[1],
                np.log(max(major, 0.5)),
                np.log(max(minor, 0.5)),
                angle,
            ]

        best_result = None
        best_inliers = None
        best_score = (-1, -np.inf)
        best_consensus_count = 0
        best_low_tilt_result = None
        best_low_tilt_subset = None
        best_low_tilt_angle = np.inf

        for subset_indices in combinations(range(points.shape[0]), 5):
            subset = points[list(subset_indices)]
            try:
                result = least_squares(
                    lambda parameters: ellipse_residual(parameters, subset),
                    initial_guess(subset),
                    bounds=bounds,
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=1000,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue

            if not result.success:
                continue

            residuals = ellipse_residual(result.x, points)
            inliers = np.abs(residuals) <= 1.5
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < 5:
                continue

            median_error = float(np.median(np.abs(residuals[inliers])))
            score = (inlier_count, -median_error)

            major_axis = np.exp(result.x[2])
            minor_axis = np.exp(result.x[3])
            axis_ratio = min(major_axis, minor_axis) / max(
                major_axis, minor_axis
            )
            inferred_tilt = float(
                np.degrees(np.arccos(np.clip(axis_ratio, 0.0, 1.0)))
            )
            if inferred_tilt < best_low_tilt_angle:
                best_low_tilt_result = result
                best_low_tilt_subset = subset_indices
                best_low_tilt_angle = inferred_tilt

            if score > best_score:
                best_result = result
                best_inliers = inliers
                best_score = score
                best_consensus_count = inlier_count

        if best_result is None:
            raise ValueError(
                "Fewer than five spoke picks fit the ring. Check the "
                "center and ring selection, then retry."
            )

        if best_consensus_count < 6:
            if best_low_tilt_result is None:
                raise ValueError(
                    "Could not fit an ellipse to any five-spoke subset."
                )
            best_result = best_low_tilt_result
            best_inliers = np.zeros(points.shape[0], dtype=bool)
            best_inliers[list(best_low_tilt_subset)] = True

        for _ in range(3):
            result = least_squares(
                lambda parameters: ellipse_residual(
                    parameters, points[best_inliers]
                ),
                best_result.x,
                bounds=bounds,
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=2000,
            )
            if not result.success:
                raise ValueError("Could not refine the consensus ellipse fit.")

            residuals = ellipse_residual(result.x, points)
            updated_inliers = np.abs(residuals) <= 1.5
            if np.count_nonzero(updated_inliers) < 5:
                break
            best_result = result
            if np.array_equal(updated_inliers, best_inliers):
                break
            best_inliers = updated_inliers

        return best_result.x[0], best_result.x[1], best_inliers

    def run_eight_spoke_calibration(self):
        """Detect the first ring on eight spokes, then calibrate and expand."""
        if self.image_data is None:
            self.set_error_status("Error: Load an image first.")
            return

        original_refine = [
            self.inputs[key][1].isChecked()
            for key in FIT_KEYS
        ]
        original_rings = self.tracked_rings
        original_points_x = list(self.tracked_points_x)
        original_points_y = list(self.tracked_points_y)
        original_calibration = self.calibrated_geometry
        original_parameters = {
            key: self.inputs[key][0].text()
            for key in FIT_KEYS
        }

        try:
            center_x = float(
                self.inputs["Beam Center X (px) "][0].text()
            )
            center_y = float(
                self.inputs["Beam Center Y (px) "][0].text()
            )
            inner_radius = float(self.txt_inner_edge.text())
            outer_radius = float(self.txt_outer_edge.text())
            lattice = float(self.txt_lattice.text())
            wavelength = float(self.txt_wave.text())
            pixel_size_x = float(self.txt_pixel_x.text())
            pixel_size_y = float(self.txt_pixel_y.text())

            values = [
                center_x, center_y, inner_radius, outer_radius,
                lattice, wavelength, pixel_size_x, pixel_size_y,
            ]
            if not np.all(np.isfinite(values)):
                raise ValueError("Center and search inputs must be finite.")
            if not 0 <= inner_radius < outer_radius:
                raise ValueError(
                    "Require a nonnegative inner radius below the outer radius."
                )

            height, width = self.image_data.shape
            if not (0 <= center_x < width and 0 <= center_y < height):
                raise ValueError(
                    "The estimated beam center must be inside the image."
                )

            calibrant = self.combo_calibrant.currentText()
            hkl = self.combo_reflection.currentData()
            if hkl is None:
                raise ValueError("Select the first-ring reflection to trace.")

            h, k, l = hkl
            if CALIBRANTS[calibrant]["system"] == "Cubic":
                d_spacing = lattice / np.sqrt(h * h + k * k + l * l)
            else:
                if h != 0 or k != 0 or l <= 0:
                    raise ValueError(
                        "The layered model supports only (00l), l > 0."
                    )
                d_spacing = lattice / l
            bragg_argument = wavelength / (2.0 * d_spacing)
            if not 0 < bragg_argument < 1:
                raise ValueError("The selected reflection is inaccessible.")

            data = np.asarray(self.image_data, dtype=float)
            points_x = []
            points_y = []
            radial_step = 1.0

            for angle in np.radians(np.arange(0.0, 360.0, 45.0)):
                unit_x, unit_y = np.cos(angle), np.sin(angle)
                limits = []
                if unit_x > 1e-12:
                    limits.append((width - 1 - center_x) / unit_x)
                elif unit_x < -1e-12:
                    limits.append(-center_x / unit_x)
                if unit_y > 1e-12:
                    limits.append((height - 1 - center_y) / unit_y)
                elif unit_y < -1e-12:
                    limits.append(-center_y / unit_y)

                ray_end = min(outer_radius, min(limits))
                radii = np.arange(
                    inner_radius, ray_end, radial_step
                )
                if radii.size < 5:
                    continue

                profile = self.sample_display_profile(
                    data,
                    center_x + radii * unit_x,
                    center_y + radii * unit_y,
                )
                background = self.estimate_radial_background(profile)
                corrected = profile - background
                smooth, peaks = self.detect_radial_peaks(
                    corrected,
                    radial_step,
                    use_global_prominence=False,
                )
                if peaks.size == 0:
                    continue

                peak_heights = smooth[peaks]
                strongest = float(np.max(peak_heights))
                if strongest <= 0:
                    continue
                eligible = peaks[peak_heights >= 0.5 * strongest]
                if eligible.size == 0:
                    continue

                peak = int(eligible[0])
                offset = 0.0
                if 0 < peak < len(smooth) - 1:
                    left, middle, right = smooth[peak - 1:peak + 2]
                    denominator = left - 2.0 * middle + right
                    if denominator != 0:
                        offset = float(np.clip(
                            0.5 * (left - right) / denominator,
                            -0.5,
                            0.5,
                        ))

                radius = radii[peak] + offset * radial_step
                points_x.append(center_x + radius * unit_x)
                points_y.append(center_y + radius * unit_y)

            self.spoke_pick_scatter.setData(x=points_x, y=points_y)
            self.plot_image()
            if len(points_x) < 5:
                raise ValueError(
                    f"Only {len(points_x)} of 8 spokes found a peak; "
                    "at least five are needed to fit an ellipse."
                )

            fitted_center_x, fitted_center_y, inliers = (
                self.fit_spoke_ellipse(points_x, points_y)
            )
            if not (
                0 <= fitted_center_x < width
                and 0 <= fitted_center_y < height
            ):
                raise ValueError(
                    "The fitted ring center is outside the image. "
                    "Check the spoke picks and initial center."
                )

            seed_x = np.asarray(points_x, dtype=float)[inliers]
            seed_y = np.asarray(points_y, dtype=float)[inliers]
            seed = {
                "calibrant": calibrant,
                "hkl": tuple(hkl),
                "x": seed_x,
                "y": seed_y,
            }

            self.inputs["Beam Center X (px) "][0].setText(
                f"{fitted_center_x:.6f}"
            )
            self.inputs["Beam Center Y (px) "][0].setText(
                f"{fitted_center_y:.6f}"
            )
            self.calibrated_geometry = None
            self.tracked_rings = [seed]
            self.tracked_points_x = seed_x.tolist()
            self.tracked_points_y = seed_y.tolist()
            self.update_ring_summary()
            self.plot_image()
            self.lbl_status.setText(
                f"Status: Picked {len(points_x)} of 8 spoke peaks; "
                f"ellipse fit retained {np.count_nonzero(inliers)}. "
                "Yellow markers show all detected spoke picks."
            )
            QtWidgets.QApplication.processEvents()

            # Ensure center and distance are refined for the bootstrap fit.
            for index, key in enumerate(FIT_KEYS):
                self.inputs[key][1].setChecked(
                    original_refine[index] or index < 3
                )

            self.run_gsas2_optimization_refinement()

        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.tracked_rings = original_rings
            self.tracked_points_x = original_points_x
            self.tracked_points_y = original_points_y
            self.calibrated_geometry = original_calibration
            for key, value in original_parameters.items():
                self.inputs[key][0].setText(value)
            self.update_ring_summary()
            self.plot_image()
            self.set_error_status(f"Error: {exc}")

        finally:
            for index, key in enumerate(FIT_KEYS):
                self.inputs[key][1].setChecked(original_refine[index])

    # Expansion/refinement controller

    def run_gsas2_optimization_refinement(self):
        """Bootstrap from the first pick, expand, and jointly refine."""
        if self.image_data is None or not self.tracked_rings:
            self.set_error_status(
                "Error: Load an image and manually pick the ring-to-trace, by default the first ring, but doublecheck the ring-to-trace selection on the right panel."
            )
            return

        keys = FIT_KEYS

        original_refine = [
            self.inputs[key][1].isChecked()
            for key in keys
        ]

        try:
            minimum_d = float(self.txt_min_d.text())
            lattice = float(self.txt_lattice.text())
            wavelength = float(self.txt_wave.text())
            pixel_size_x = float(self.txt_pixel_x.text())
            pixel_size_y = float(self.txt_pixel_y.text())

            values = [
                minimum_d, lattice, wavelength,
                pixel_size_x, pixel_size_y,
            ]

            if not np.all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError(
                    "Minimum d, lattice, wavelength, and pixel sizes "
                    "must be finite and positive."
                )

            name = self.combo_calibrant.currentText()

            # The first stored ring remains the manual anchor.
            # Previously expanded rings are discarded on every rerun.
            seed = self.tracked_rings[0]

            if seed["calibrant"] != name:
                raise ValueError(
                    "Clear the rings and pick the ring-to-trace, by default the first ring, "
                    "for the selected calibrant. Doublecheck the ring-to-trace selection."
                )

            h, k, l = seed["hkl"]

            if CALIBRANTS[name]["system"] == "Cubic":
                seed_d = lattice / np.sqrt(h*h + k*k + l*l)
            else:
                seed_d = lattice / l

            argument = wavelength / (2 * seed_d)

            if not 0 < argument < 1:
                raise ValueError("The seed reflection is inaccessible.")

            seed_angle = 2 * np.arcsin(argument)

            if seed_angle >= np.pi / 2:
                raise ValueError(
                    "The current forward geometry supports "
                    "forward-scattering rings only."
                )

            candidates = self.expansion_reflections(minimum_d, lattice)

            # Start from a stable untilted approximation.
            geometry = self.current_geometry()
            radius_mm = np.median(np.hypot(
                (seed["x"] - geometry[0]) * pixel_size_x,
                (seed["y"] - geometry[1]) * pixel_size_y,
            ))

            if original_refine[2]:
                estimate = radius_mm / np.tan(seed_angle)
                self.inputs[keys[2]][0].setText(f"{estimate:.8g}")

            if original_refine[3]:
                self.inputs[keys[3]][0].setText("0.0")

            # Bootstrap only selected center/distance variables.
            # A single ring is a weak basis for simultaneous tilt fitting.
            # A single ring does not independently determine wavelength and distance.
            # A single ring cannot reliably distinguish distance from penetration.
            for index in (3, 4, 5, 6):
                self.inputs[keys[index]][1].setChecked(False)

            self.tracked_rings = [seed]

            if any(original_refine[:3]):
                if not self._refine_collected_rings():
                    return

            for key, checked in zip(keys, original_refine):
                self.inputs[key][1].setChecked(checked)

            # Avoid starting joint tilt/azimuth fitting at exactly zero tilt.
            if original_refine[3] and original_refine[4]:
                self.inputs[keys[3]][0].setText("0.2")

            for iteration in range(3):
                # Use the wavelength produced by the previous joint fit.
                wavelength = float(self.txt_wave.text())

                seed_argument = wavelength / (2.0 * seed_d)
                if not 0 < seed_argument < 1:
                    raise ValueError(
                        "The seed reflection is inaccessible at the fitted wavelength."
                    )
                seed_target = 2.0 * np.arcsin(seed_argument)

                if iteration == 0:
                    # The manual points have served as the bootstrap fit.
                    self.tracked_points_x = []
                    self.tracked_points_y = []
                    self.plot_image()

                self.lbl_status.setText(
                    f"Status: Resampling seed ring {seed['hkl']}..."
                )
                QtWidgets.QApplication.processEvents()

                traced_seed = self.trace_predicted_ring(seed_target)
                if traced_seed is None:
                    raise ValueError(
                        "Could not resample the selected seed ring from "
                        "the fitted geometry. Check the center, distance, "
                        "and ring assignment, then retry."
                    )

                seed_x, seed_y = traced_seed
                seed = {
                    "calibrant": name,
                    "hkl": tuple(seed["hkl"]),
                    "x": seed_x,
                    "y": seed_y,
                }
                collected = [seed]

                for hkl, d_spacing in candidates:
                    # Search outward from the manually selected anchor.
                    if d_spacing >= seed_d * (1 - 1e-10):
                        continue

                    argument = wavelength / (2 * d_spacing)
                    if not 0 < argument < 1:
                        continue

                    target = 2 * np.arcsin(argument)

                    if target >= np.pi / 2:
                        continue

                    self.lbl_status.setText(
                        f"Status: Expansion pass {iteration + 1}, "
                        f"searching {hkl}, d={d_spacing:.4g} Å..."
                    )
                    QtWidgets.QApplication.processEvents()

                    traced = self.trace_predicted_ring(target)

                    if traced is None:
                        continue

                    x, y = traced
                    collected.append({
                        "calibrant": name,
                        "hkl": tuple(hkl),
                        "x": x,
                        "y": y,
                    })

                self.tracked_rings = collected
                self.refresh_expanded_ring_display()

                if len(collected) < 2:
                    self.lbl_status.setText(
                        "Status: Seed retained, but no additional rings "
                        "passed detection. Check geometry and ring identity."
                    )
                    return

                if not self._refine_collected_rings():
                    return

            self.lbl_status.setText(
                f"Status: Expansion/refinement complete. "
                f"{len(self.tracked_rings)} rings collected; "
                f"expansion cutoff d={minimum_d:.4g} Å."
            )

        except (ValueError, FloatingPointError) as exc:
            self.set_error_status(f"Error: {exc}")

        finally:
            for key, checked in zip(keys, original_refine):
                self.inputs[key][1].setChecked(checked)

            self.refresh_expanded_ring_display()

    def refresh_expanded_ring_display(self):
        if self.tracked_rings:
            self.tracked_points_x = np.concatenate(
                [ring["x"] for ring in self.tracked_rings]
            ).tolist()
            self.tracked_points_y = np.concatenate(
                [ring["y"] for ring in self.tracked_rings]
            ).tolist()
        else:
            self.tracked_points_x = []
            self.tracked_points_y = []

        self.update_ring_summary()
        self.plot_image()

    # This runs three successive fits per click, starting each fit from the parameters produced by the previous fit.

    def current_fit_parameters(self):
        """Return geometry plus wavelength for refinement reporting."""
        values = np.array([
            float(self.inputs[key][0].text())
            for key in FIT_KEYS
        ])
    
        if not np.all(np.isfinite(values)):
            raise ValueError("Fit parameters must be finite.")
    
        # Reuse the geometry validation already implemented.
        self.current_geometry()
    
        if values[-1] <= 0:
            raise ValueError("Wavelength must be positive.")
    
        return values

    def reoptimize_current_rings(self):
        """Resample all stored rings from current geometry, then refit."""
        if self.image_data is None:
            self.lbl_status.setText(
                "Status: Load an image before recalibrating."
            )
            return

        had_original_picks = bool(self.tracked_rings)
        param_keys = FIT_KEYS

        if not any(
            self.inputs[key][1].isChecked()
            for key in param_keys
        ):
            self.set_error_status(
                "Error: Select at least one parameter for refinement."
            )
            return

        try:
            before = self.current_fit_parameters()
        except ValueError as exc:
            self.set_error_status(f"Error: {exc}")
            return

        original_rings = self.tracked_rings
        original_points_x = list(self.tracked_points_x)
        original_points_y = list(self.tracked_points_y)
        original_calibration = self.calibrated_geometry
        original_parameters = {
            key: self.inputs[key][0].text()
            for key in FIT_KEYS
        }

        cycles = 3

        # Disable actions while the optimizer processes Qt events.
        buttons = (
            self.btn_load_image,
            self.btn_pick_center,
            self.btn_spoke_calibrate,
            self.btn_refine,
            self.btn_reoptimize,
            self.btn_clear,
            self.btn_export_imctrl,
            self.btn_save_pipeline,
        )
        previous_enabled = [
            button.isEnabled() for button in buttons
        ]

        for button in buttons:
            button.setEnabled(False)

        try:
            calibrant = self.combo_calibrant.currentText()
            system = CALIBRANTS[calibrant]["system"]
            lattice = float(self.txt_lattice.text())

            if not had_original_picks:
                minimum_d = float(self.txt_min_d.text())
                if not np.isfinite(minimum_d) or minimum_d <= 0:
                    raise ValueError(
                        "Minimum d-spacing must be finite and positive."
                    )
                candidates = self.expansion_reflections(
                    minimum_d, lattice
                )
                self.tracked_rings = [
                    {
                        "calibrant": calibrant,
                        "hkl": tuple(hkl),
                        "x": np.empty(0, dtype=float),
                        "y": np.empty(0, dtype=float),
                    }
                    for hkl, _ in candidates
                ]
                if not self.tracked_rings:
                    raise ValueError(
                        "No candidate reflections are available for this "
                        "calibrant and minimum d-spacing."
                    )
                self.lbl_status.setText(
                    "Status: No stored ring picks; searching predicted "
                    "calibrant rings from the current geometry..."
                )
                QtWidgets.QApplication.processEvents()

            for cycle in range(cycles):
                wavelength = float(self.txt_wave.text())
                resampled_rings = []

                for ring_index, ring in enumerate(
                    self.tracked_rings, start=1
                ):
                    if ring["calibrant"] != calibrant:
                        raise ValueError(
                            "Stored rings belong to a different calibrant. "
                            "Select the original standard before recalibrating."
                        )

                    h, k, l = ring["hkl"]
                    if system == "Cubic":
                        d_spacing = lattice / np.sqrt(
                            h * h + k * k + l * l
                        )
                    else:
                        if h != 0 or k != 0 or l <= 0:
                            raise ValueError(
                                "The layered model supports only (00l), l > 0."
                            )
                        d_spacing = lattice / l

                    bragg_argument = wavelength / (2.0 * d_spacing)
                    if not 0 < bragg_argument < 1:
                        if not had_original_picks:
                            continue
                        raise ValueError(
                            f"Ring {ring_index}, reflection {ring['hkl']}: "
                            "inaccessible at the current wavelength."
                        )

                    target_angle = 2.0 * np.arcsin(bragg_argument)
                    if target_angle >= np.pi / 2:
                        if not had_original_picks:
                            continue
                        raise ValueError(
                            f"Ring {ring_index}, reflection {ring['hkl']}: "
                            "outside the forward-scattering range."
                        )

                    self.lbl_status.setText(
                        f"Status: Recalibration cycle {cycle + 1}/{cycles}, "
                        f"resampling ring {ring_index}/{len(self.tracked_rings)} "
                        f"{ring['hkl']}..."
                    )
                    QtWidgets.QApplication.processEvents()

                    traced = self.trace_predicted_ring(target_angle)
                    if traced is None:
                        if not had_original_picks:
                            continue
                        raise ValueError(
                            f"Could not resample ring {ring_index}, "
                            f"reflection {ring['hkl']}. Check the geometry "
                            "and ring assignment, then retry."
                        )

                    points_x, points_y = traced
                    resampled_rings.append({
                        "calibrant": calibrant,
                        "hkl": tuple(ring["hkl"]),
                        "x": points_x,
                        "y": points_y,
                    })

                if not resampled_rings:
                    raise ValueError(
                        "No candidate rings could be resampled from the "
                        "current geometry. Check the center, distance, "
                        "calibrant, and wavelength."
                    )

                self.tracked_rings = resampled_rings
                self.refresh_expanded_ring_display()
                self.lbl_status.setText(
                    f"Status: Fitting {len(self.tracked_rings)} "
                    f"resampled rings, cycle {cycle + 1}/{cycles}..."
                )
                QtWidgets.QApplication.processEvents()

                if not self._refine_collected_rings():
                    raise ValueError(
                        "Joint fit failed after resampling."
                    )

            after = self.current_fit_parameters()
            changes = after - before
            
            # Avoid reporting a nearly 360-degree change across the wrap boundary.
            changes[4] = (changes[4] + 180.0) % 360.0 - 180.0

            # Keep the final fit's RMS information in the status message.
            fit_status = self.lbl_status.text()
            self.lbl_status.setText(
                f"{fit_status} "
                #f"Completed {cycles} reoptimization cycles. "
                #"Ring picks unchanged.\n"
                #"Parameter changes: "
                f"\u0394X={changes[0]:+.3g} px, "
                f"\u0394Y={changes[1]:+.3g} px, "
                f"\u0394dist={changes[2]:+.3g} mm, "
                f"\u0394tilt={changes[3]:+.3g}°, "
                f"\u0394rot={changes[4]:+.3g}°, "
                f"\u0394pen={changes[5]:+.3g}, "
                f"\u0394λ={changes[6]:+.3g} Å."
            )

        except (ValueError, FloatingPointError) as exc:
            self.tracked_rings = original_rings
            self.tracked_points_x = original_points_x
            self.tracked_points_y = original_points_y
            self.calibrated_geometry = original_calibration
            for key, value in original_parameters.items():
                self.inputs[key][0].setText(value)
            self.btn_export_imctrl.setEnabled(
                original_calibration is not None
            )
            self.btn_save_pipeline.setEnabled(
                self.pipeline_handoff_file is not None
                and original_calibration is not None
            )
            self.update_ring_summary()
            self.plot_image()
            if had_original_picks:
                self.set_error_status(
                    f"Error: {exc} Original rings and parameters restored."
                )
            else:
                self.lbl_status.setText(
                    f"Status: Recalibration could not proceed: {exc} "
                    "Current parameters and ring state were retained."
                )

        finally:
            for button, enabled in zip(
                buttons, previous_enabled
            ):
                button.setEnabled(enabled)

    # shared sampling and peak-detection helpers

    def sample_display_profile(self, data, sample_x, sample_y):
        """Sample array intensities at default ImageItem display coordinates."""
        shift = 0.5 if self.chk_pixel_centers.isChecked() else 0.0

        return map_coordinates(
            data,
            [np.asarray(sample_y) - shift,
             np.asarray(sample_x) - shift],
            order=1,
            mode="nearest",
            prefilter=False,
        )

    def detect_radial_peaks(
        self, profile, radial_step, use_global_prominence=True
    ):
        """Smooth a radial profile and identify sufficiently prominent peaks."""
        sigma_px = self.spin_smooth_sigma.value()

        if sigma_px > 0:
            smooth = gaussian_filter1d(
                profile,
                sigma=sigma_px / radial_step,
            )
        else:
            smooth = profile.copy()

        differences = np.diff(profile)

        noise = (
            1.4826
            * np.median(
                np.abs(differences - np.median(differences))
            )
            / np.sqrt(2.0)
        )

        prominence = self.spin_noise_multiplier.value() * noise
        if use_global_prominence:
            prominence = max(
                prominence,
                self.spin_prominence_fraction.value()
                * float(np.ptp(smooth)),
            )
        prominence = max(prominence, 1e-12)

        separation_samples = max(
            1,
            int(np.ceil(
                self.spin_peak_separation.value() / radial_step
            )),
        )

        peaks, _ = find_peaks(
            smooth,
            prominence=prominence,
            distance=separation_samples,
        )

        return smooth, peaks

    def estimate_radial_background(
        self, profile, asymmetry=0.01, smoothness=1e5, iterations=10
    ):
        """Estimate a smooth lower-envelope background with asymmetric LS."""
        profile = np.asarray(profile, dtype=float)
        sample_count = profile.size
        if sample_count < 3:
            return profile.copy()

        second_difference = sparse.diags(
            [
                np.ones(sample_count - 2),
                -2.0 * np.ones(sample_count - 2),
                np.ones(sample_count - 2),
            ],
            [0, 1, 2],
            shape=(sample_count - 2, sample_count),
            format="csc",
        )
        penalty = smoothness * (second_difference.T @ second_difference)
        weights = np.ones(sample_count, dtype=float)

        for _ in range(iterations):
            weight_matrix = sparse.diags(weights, format="csc")
            background = spsolve(
                weight_matrix + penalty,
                weights * profile,
            )
            weights = np.where(
                profile > background,
                asymmetry,
                1.0 - asymmetry,
            )

        return background
        
    def scattering_coordinates_at(self, x, y):
        """Return calibrated (2theta_deg, azimuth_deg, d_A, q_inv_A).
    
        Coordinates x and y are ImageItem display coordinates, matching
        the coordinates stored by the ring-tracing methods.
    
        Azimuth convention:
            0 degrees along displayed +X,
            90 degrees along displayed +Y,
            wrapped into [0, 360).
    
        Azimuth is evaluated in the transverse scattering plane after
        applying the same detector-tilt transformation as the forward model.
        """
        calibration = self.calibrated_geometry
        if calibration is None:
            return None
    
        parameters = calibration["parameters"]
        pixel_size_x = calibration["pixel_size_x"]
        pixel_size_y = calibration["pixel_size_y"]
        wavelength = calibration["wavelength"]
    
        cx, cy, distance, tilt, tilt_rotation = parameters[:5]
    
        # The original projection function has no penetration argument.
        # Supply it only for the six-parameter implementation.
        projection_options = {}
        if parameters.size >= 6:
            projection_options["penetration"] = float(parameters[5])
    
        two_theta = float(
            gsas2_metric_projection(
                np.array([x], dtype=float),
                np.array([y], dtype=float),
                cx * pixel_size_x,
                cy * pixel_size_y,
                distance,
                tilt,
                tilt_rotation,
                pixel_size_x,
                pixel_size_y,
                **projection_options,
            )[0]
        )
    
        # Reproduce the forward model's transverse-vector transformation.
        dx = (x - cx) * pixel_size_x
        dy = (y - cy) * pixel_size_y
    
        tilt_rad = np.radians(tilt)
        azimuth_rad = np.radians(tilt_rotation - 90.0)
        cos_p = np.cos(azimuth_rad)
        sin_p = np.sin(azimuth_rad)
    
        x_rot = dx * cos_p + dy * sin_p
        y_rot = -dx * sin_p + dy * cos_p
    
        # Tilt scales the component parallel to the tilt direction.
        # Rotate back to the displayed X/Y reference orientation.
        transverse_x = x_rot * np.cos(tilt_rad)
        lab_x = transverse_x * cos_p - y_rot * sin_p
        lab_y = transverse_x * sin_p + y_rot * cos_p
    
        transverse_radius = np.hypot(lab_x, lab_y)
    
        # Azimuth is undefined exactly at the beam center.
        if transverse_radius <= 1e-12:
            azimuth_deg = np.nan
        else:
            azimuth_deg = float(
                np.degrees(np.arctan2(lab_y, lab_x)) % 360.0
            )
    
        if not np.isfinite(two_theta) or two_theta < 0:
            return np.nan, azimuth_deg, np.nan, np.nan
    
        sin_theta = np.sin(0.5 * two_theta)
    
        if sin_theta <= 1e-15:
            d_spacing = np.inf
            q_value = 0.0
        else:
            d_spacing = wavelength / (2.0 * sin_theta)
            q_value = 4.0 * np.pi * sin_theta / wavelength
    
        return (
            float(np.degrees(two_theta)),
            azimuth_deg,
            float(d_spacing),
            float(q_value),
        )

    def update_beam_center_marker(self):
        """Display the configured center estimate or calibrated center."""
        calibration = self.calibrated_geometry

        if self.image_data is None:
            self.beam_center_horizontal.hide()
            self.beam_center_vertical.hide()
            return

        if calibration is not None:
            # Fitted center coordinates already use ImageItem display coordinates.
            # Do not add or subtract the interpolation half-pixel correction.
            cx, cy = calibration["parameters"][:2]
        else:
            try:
                cx = float(
                    self.inputs["Beam Center X (px) "][0].text()
                )
                cy = float(
                    self.inputs["Beam Center Y (px) "][0].text()
                )
            except ValueError:
                self.beam_center_horizontal.hide()
                self.beam_center_vertical.hide()
                return
    
        if not np.all(np.isfinite([cx, cy])):
            self.beam_center_horizontal.hide()
            self.beam_center_vertical.hide()
            return
    
        half_length = 7.0
        self.beam_center_horizontal.setData(
            x=[float(cx) - half_length, float(cx) + half_length],
            y=[float(cy), float(cy)],
        )
        self.beam_center_vertical.setData(
            x=[float(cx), float(cx)],
            y=[float(cy) - half_length, float(cy) + half_length],
        )
        self.beam_center_horizontal.show()
        self.beam_center_vertical.show()

# ==============================================================================
# RUNTIME EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline_handoff_file = None
    image_filename = None
    arguments = iter(sys.argv[1:])
    for argument in arguments:
        if argument == "--pipeline-handoff":
            pipeline_handoff_file = next(arguments, None)
        elif image_filename is None:
            image_filename = argument

    app = QtWidgets.QApplication(sys.argv)
    window = GSAS2CalibratorApp(
        pipeline_handoff_file=pipeline_handoff_file
    )
    window.show()

    # Optional: provide a TIFF filename on the command line.
    if image_filename is not None:
        window.load_image(image_filename)

    sys.exit(app.exec())
