import pyqtgraph as pg
import os
import glob
import numpy as np

from mainUI.UI_settings import Settings


class CSimView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, settings: Settings):
        super().__init__(parent)
        # global tiflist, keylist, curr_key, curr_pos
        self.setBackground("w")
        # self.directory = directory
        self.settings = settings
        # similar to contour view, show full directory
        self.view = self.addPlot(title="")
        self.min = 0
        self.max = 100
        self.spacing = 1
        self.methods = (
            "Compared to First",
            "Compared to Previous",
        )
        self.similarity_line = {}
        self.similarity_line_data = []
        self.legend = self.view.addLegend(offset=(-1, 1))
        for k in self.methods:
            # check if addPlot.plot() is a shortcut for create PlotItem, add PlotItem
            self.similarity_line[k] = self.view.plot()
            self.legend.addItem(self.similarity_line[k], k)
        self.similarity_line["Compared to Previous"].setPen("r")
        self.similarity_line["Compared to First"].setPen('b')

    def update_dir(self):
        self.update_data()

    def update_data(self):
        filename_piece = os.path.join(
            self.settings.output_directory,
            "stats",
            self.settings.keylist[self.settings.curr_key] + "*_csim.txt"
        )
        # print(f"{filename_piece = }")
        filenames = glob.glob(
            filename_piece
        )
        # print(f"CSim filenames: {filenames}")
        filenames.sort(key = lambda x: (len(x), x))
        arrays = [np.loadtxt(filename) for filename in filenames]
        self.similarity_line_data = np.vstack(arrays)
        for i, k in enumerate(self.methods):
            # [:,0] for comparison to first, [:,1] for comparison to previous
            self.similarity_line[k].setData(self.similarity_line_data[:, i])


