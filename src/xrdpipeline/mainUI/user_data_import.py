"""
XRDdatapipeline is a package for automated XRD data masking and integration.
Copyright (C) 2025 UChicago Argonne, LLC
Full copyright info can be found in the LICENSE included with this project or at
https://github.com/AdvancedPhotonSource/XRDdatapipeline/blob/main/LICENSE

This file defines the user data import widget for the results UI.
"""


import numpy as np
from dataclasses import dataclass, field
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import os

from mainUI.UI_settings import ColorSettings, Settings

@dataclass
class UserAddedDataInstance:
    name: str
    file_name: str
    data: np.array
    plotitem: pg.PlotDataItem = field(init=False)
    x_type: str = "tth"
    color: ColorSettings = field(init=False)
    offset: int = 0
    multiplier: float = 1

    def __post_init__(self):
        self.color = ColorSettings(self.name, color="black")
        self.plotitem = pg.PlotDataItem()
        self.plotitem.setData(self.data)
        self.plotitem.setPen("black")


def import_data_instance(file_name, name, x_type):
    array_data = np.loadtxt(file_name)
    return UserAddedDataInstance(name=name, file_name=file_name, data=array_data, x_type=x_type)

locations = {
    0: "Unset",
    1: "Integral",
    2: "Stats",
}

x_types = ["tth", "Q"]

class UserAddedDataInstanceWidget(QtWidgets.QWidget):
    delete_userdata_signal = pg.QtCore.Signal(QtWidgets.QWidget)

    def __init__(self, parent, data_instance: UserAddedDataInstance):
        super().__init__(parent)
        self.row_layout = QtWidgets.QHBoxLayout()

        self.data_instance = data_instance
        self.graph = []
        self.color = data_instance.color
        self.offset = data_instance.offset
        self.multiplier = data_instance.multiplier
        
        self.name_field = QtWidgets.QLineEdit(self.data_instance.name)
        self.filename_field = QtWidgets.QLabel(self.data_instance.file_name)
        self.color_field = QtWidgets.QLineEdit(self.color.color)
        self.offset_field = QtWidgets.QSpinBox()
        self.multiplier_field = QtWidgets.QSpinBox()
        self.multiplier_field.setValue(self.multiplier)
        self.multiplier_field.setMaximum(1000000)
        self.x_type_field = QtWidgets.QComboBox()
        self.x_type_field.addItems(x_types)
        self.location_field = QtWidgets.QComboBox()
        self.location_field.addItems(locations.values())
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.released.connect(self.remove_button_pressed)

        self.filename_scroll = QtWidgets.QScrollArea()
        self.filename_scroll.setWidgetResizable(True)
        self.filename_scroll.setWidget(self.filename_field)

        self.row_layout.addWidget(self.name_field, 1)
        self.row_layout.addWidget(self.filename_scroll, 3)
        self.row_layout.addWidget(self.color_field, 1)
        self.row_layout.addWidget(self.offset_field, 1)
        self.row_layout.addWidget(self.multiplier_field, 1)
        self.row_layout.addWidget(self.x_type_field, 1)
        self.row_layout.addWidget(self.location_field, 1)
        self.row_layout.addWidget(self.remove_button, 1)
        self.setLayout(self.row_layout)
    
    def remove_button_pressed(self):
        self.delete_userdata_signal.emit(self)


class UserAddedDataTab(QtWidgets.QWidget):
    update_userdata_signal = pg.QtCore.Signal(UserAddedDataInstance,str)
    remove_deleted_userdata_from_plot = pg.QtCore.Signal(UserAddedDataInstance)

    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        self.settings = settings
        self.main_layout = QtWidgets.QGridLayout()

        self.user_data_table_visual = QtWidgets.QWidget()
        self.user_data_table_layout = QtWidgets.QVBoxLayout()
        self.user_data_titlebar = QtWidgets.QWidget()
        self.user_data_titlebarlayout = QtWidgets.QHBoxLayout()
        self.user_data_namelabel = QtWidgets.QLabel("Name")
        self.user_data_filenamelabel = QtWidgets.QLabel("Filename")
        self.user_data_colorlabel = QtWidgets.QLabel("Color")
        self.user_data_offsetlabel = QtWidgets.QLabel("Offset")
        self.user_data_multiplierlabel = QtWidgets.QLabel("Multiplier")
        self.user_data_xtypelabel = QtWidgets.QLabel("X axis type")
        self.user_data_locationlabel = QtWidgets.QLabel("Display")
        self.user_data_removelabel = QtWidgets.QLabel("Remove")
        self.user_data = []
        self.user_data_table_visual.setLayout(self.user_data_table_layout)

        self.user_data_table_scroll = QtWidgets.QScrollArea()
        self.user_data_table_scroll.setWidgetResizable(True)
        self.user_data_table_scroll.setWidget(self.user_data_table_visual)

        self.add_new_button = QtWidgets.QPushButton("Import new data")
        self.add_new_button.released.connect(self.add_new)

        self.update_button = QtWidgets.QPushButton("Update")
        self.update_button.released.connect(self.send_update)

        self.user_data_titlebarlayout.addWidget(self.user_data_namelabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_filenamelabel, 3)
        self.user_data_titlebarlayout.addWidget(self.user_data_colorlabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_offsetlabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_multiplierlabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_xtypelabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_locationlabel, 1)
        self.user_data_titlebarlayout.addWidget(self.user_data_removelabel, 1)
        self.user_data_titlebar.setLayout(self.user_data_titlebarlayout)

        self.main_layout.addWidget(self.user_data_titlebar)
        self.main_layout.addWidget(self.user_data_table_scroll)
        self.main_layout.addWidget(self.add_new_button)
        self.main_layout.addWidget(self.update_button)
        self.setLayout(self.main_layout)

    def add_new(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Choose File",
            ".",
            "",
        )[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        data = import_data_instance(filename, name, "tth")
        self.user_data.append(UserAddedDataInstanceWidget(parent=self.user_data_table_visual,data_instance=data))
        self.user_data_table_layout.addWidget(self.user_data[-1])
        self.user_data[-1].delete_userdata_signal.connect(self.remove)

    def remove(self, wid):
        if wid in self.user_data:
            self.user_data.remove(wid)
        self.user_data_table_layout.removeWidget(wid)
        self.remove_deleted_userdata_from_plot.emit(wid.data_instance)
        wid.deleteLater()

    def send_update(self):
        for datum in self.user_data:
            datum.data_instance.name = datum.name_field.text()
            datum.data_instance.color.color = datum.color_field.text()
            datum.data_instance.offset = datum.offset_field.value()
            datum.data_instance.multiplier = datum.multiplier_field.value()
            datum.data_instance.x_type = datum.x_type_field.currentData()
            datum.data_instance.plotitem.setData(datum.data_instance.data[:,0], datum.data_instance.data[:,1] * datum.data_instance.multiplier + datum.data_instance.offset)
            datum.data_instance.plotitem.setPen(datum.data_instance.color.color)
            self.update_userdata_signal.emit(datum.data_instance, locations[datum.location_field.currentIndex()])

    def update_dir(self):
        # clear data?
        return