import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from mainUI.UI_settings import Settings
from mainUI.stats import StatsView
from mainUI.contour import ContourView
from mainUI.csim import CSimView
from mainUI.spottiness import SpottinessView
from mainUI.user_data_import import UserAddedDataTab

class TabbedArea(QtWidgets.QTabWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings

        self.stats_page = QtWidgets.QWidget()
        self.contour_page = QtWidgets.QWidget()
        self.csim_page = QtWidgets.QWidget()
        self.spottiness_page = QtWidgets.QWidget()
        self.user_data_page = QtWidgets.QWidget()

        self.stats_widget = StatsView(self.stats_page, self.settings)
        self.contour_widget = ContourView(self.contour_page, self.settings)
        self.csim_widget = CSimView(self.csim_page, self.settings)
        self.spottiness_widget = SpottinessView(self.spottiness_page, self.settings)
        self.user_data_widget = UserAddedDataTab(self.user_data_page, self.settings)

        self.stats_layout = QtWidgets.QGridLayout()
        self.contour_layout = QtWidgets.QGridLayout()
        self.csim_layout = QtWidgets.QGridLayout()
        self.spottiness_layout = QtWidgets.QGridLayout()
        self.user_data_layout = QtWidgets.QGridLayout()

        self.stats_layout.addWidget(self.stats_widget, 0, 0, 2, 6)
        self.stats_layout.addWidget(self.stats_widget.histogram_type_select, 2, 0)

        self.stats_page.setLayout(self.stats_layout)

        self.contour_layout.addWidget(self.contour_widget, 0, 0, 5, 6)
        self.contour_layout.addWidget(self.contour_widget.live_update_checkbox, 5, 0)
        self.contour_layout.addWidget(self.contour_widget.tth_line_checkbox, 5, 1)
        self.contour_layout.addWidget(self.contour_widget.integral_select, 5, 2)
        self.contour_layout.addWidget(self.contour_widget.live_integral_min_label, 6, 0)
        self.contour_layout.addWidget(self.contour_widget.live_integral_min, 6, 1)
        self.contour_layout.addWidget(self.contour_widget.live_integral_max_label, 6, 2)
        self.contour_layout.addWidget(self.contour_widget.live_integral_max, 6, 3)
        self.contour_layout.addWidget(
            self.contour_widget.live_integral_step_label, 6, 4
        )
        self.contour_layout.addWidget(self.contour_widget.live_integral_step, 6, 5)
        self.contour_layout.addWidget(self.contour_widget.integral_min_label, 7, 0)
        self.contour_layout.addWidget(self.contour_widget.integral_min, 7, 1)
        self.contour_layout.addWidget(self.contour_widget.integral_max_label, 7, 2)
        self.contour_layout.addWidget(self.contour_widget.integral_max, 7, 3)
        self.contour_layout.addWidget(self.contour_widget.integral_step_label, 7, 4)
        self.contour_layout.addWidget(self.contour_widget.integral_step, 7, 5)
        self.contour_page.setLayout(self.contour_layout)

        self.csim_layout.addWidget(self.csim_widget)
        self.csim_page.setLayout(self.csim_layout)

        self.spottiness_layout.addWidget(self.spottiness_widget)
        self.spottiness_page.setLayout(self.spottiness_layout)

        self.user_data_layout.addWidget(self.user_data_widget)
        self.user_data_page.setLayout(self.user_data_layout)

        self.addTab(self.contour_page, "Contour")
        self.addTab(self.stats_page, "Stats")
        self.addTab(self.spottiness_page, "Spottiness")
        self.addTab(self.csim_page, "Similarity")
        self.addTab(self.user_data_page, "User Data")

    def update_dir(self):
        self.contour_widget.update_dir()
        self.stats_widget.update_dir()
        self.csim_widget.update_dir()
        self.spottiness_widget.update_dir()
        self.user_data_widget.update_dir()

