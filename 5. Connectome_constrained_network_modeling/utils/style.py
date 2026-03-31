from abc import ABC


class Style(ABC):
    palette = {"default": ["deepskyblue", "deeppink", "orange", "springgreen", "tomato", "silver", "mediumorchid",
               "mediumaquamarine", "firebrick", "blue", "red", "green", "brown"]}
    font_size = 12
    padding=2
    plot_height = 1
    plot_width = 3
    xpos_start = 0.5
    ypos_start = 0.5

    page_tight = False

    def __init__(self, **kwargs):
        pass

    def add_palette(self, label: str, palette: list):
        self.palette[label] = palette