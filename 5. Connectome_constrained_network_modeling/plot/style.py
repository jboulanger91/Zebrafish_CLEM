import copy

from matplotlib.colors import ListedColormap

from utils.style import Style
from utils.palette import Palette


class RNNDSStyle(Style):
    def __init__(self, plot_label_i=0, stimulus_palette=Palette.green_short):
        Style.__init__(self)
        self.plot_label_i = plot_label_i
        self.add_palette("stimulus", stimulus_palette)


    palette = {"default": Palette.arlecchino,
               "correct_incorrect": Palette.correct_incorrect,
               "neutral": [Palette.color_neutral],
               "green": Palette.green_short,
               "fish_code": ["#73489C", "#753B51", "#103882", "#7F0C0C"],

               "neurons_3": ["#efb233", "#67bed9", "#a18cbd"],
               "neurons_4": ["#efb233", "#de68a4", "#67bed9", "#a18cbd"]
               }
    cmap_list = {"neurons_3": ListedColormap(["#efb233", "#67bed9", "#a18cbd"]),
                 "neurons_4": ListedColormap(["#efb233", "#de68a4", "#67bed9", "#a18cbd"]),
                 "neurons_5": ListedColormap(["#efb233", "#de68a4", "#67bed9", "#a18cbd", "#909090"])}

    population_name_list = ["Left iMI", "Left cMI", "Left MON", "Left sMI",
                            "Right iMI", "Right cMI", "Right MON", "Right sMI"]

    font_size_label = 8
    font_size_text = 6
    linewidth = 1
    linewidth_single_fish = 0.05

    plot_size = 1
    plot_size_small = 0.5
    plot_size_big = 2

    padding = 1.5
    padding_small = 0.75
    padding_in_plot = 0.5
    padding_in_plot_small = 0.05

    plot_height = 1
    plot_height_small = 0.25

    plot_width = 1
    plot_width_large = 3
    plot_width_small = 0.9

    xpos_start = 1
    ypos_start = 27

    page_tight = False

    plot_label_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q" ,"r", "s", "t", "u", "v", "w", "x" ,"y", "z"]

    def add_palette(self, label: str, palette: list):
        self.palette[label] = copy.deepcopy(palette)

    def get_plot_label(self):
        label_to_show = self.plot_label_list[self.plot_label_i]
        self.plot_label_i += 1
        return label_to_show