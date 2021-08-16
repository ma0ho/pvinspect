import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pvinspect.common.transform import Transform
from pvinspect.data.image import Image
from pvinspect.data.image.show_plugin import ShowPlugin, register_show_plugin


class ShowCellCrossings(ShowPlugin):
    def __init__(self, title: str = "cell crossings", priority: int = -50):
        super().__init__(title, priority=priority)

    def apply(
        self, ax: Axes, image: Image, show_cell_crossings: bool = True, **kwargs
    ) -> None:
        """Show the crossings between cells

        Args:
            show_cell_crossings (bool): Show the cell crossings
        """
        super().apply(ax, image, **kwargs)

        if show_cell_crossings:
            x, y = np.mgrid[
                0 : image.get_meta("cols") + 1 : 1, 0 : image.get_meta("rows") + 1 : 1
            ]
            grid = np.stack([x.flatten(), y.flatten()], axis=1)
            coords = image.get_meta("transform").__call__(grid)
            ax.scatter(coords[:, 0], coords[:, 1], c="yellow", marker="+")

    def is_active(self, image: Image) -> bool:
        super().is_active(image)
        return (
            image.has_meta("transform")
            and isinstance(image.get_meta("transform"), Transform)
            and image.has_meta("rows")
            and image.has_meta("cols")
        )


class ShowMultimoduleDetections(ShowPlugin):
    def __init__(self, title: str = "multimodule detection", priority: int = -50):
        super().__init__(title, priority=priority)

    def apply(
        self,
        ax: Axes,
        image: Image,
        multimodule_show_numbers: bool = True,
        multimodule_highlight_selection: bool = True,
        multimodule_numbers_fontsize: int = 20,
        multimodule_show_boxes: bool = True,
        multimodule_boxes_linewidth: int = 2,
        **kwargs
    ) -> None:
        """Highlight detected modules from multimodule detection

        Args:
            multimodule_show_numbers (bool): Show the numbers on original multimodule images after detection
            multimodule_highlight_selection (bool): Highlight the current instance on the multimodule image
            multimodule_numbers_fontsize (int): Fonsize of multimodule numbes
            multimodule_show_boxes (bool): Show boxes surrounding detected modules
            multimodule_boxes_linewidth (bool): Line width of displayed boxes
        """
        super().apply(ax, image, **kwargs)

        if (
            multimodule_show_boxes
            and isinstance(image, Image)
            and image.has_meta("multimodule_boxes")
        ):
            for i, box in enumerate(image.get_meta("multimodule_boxes")):
                color = (
                    "red"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "yellow"
                )
                ax.plot(
                    *box[1].exterior.xy,
                    linewidth=multimodule_boxes_linewidth,
                    color=color,
                )

        if (
            multimodule_show_numbers
            and isinstance(image, Image)
            and image.has_meta("multimodule_boxes")
        ):
            for i, box in enumerate(image.get_meta("multimodule_boxes")):
                bgcolor = (
                    "red"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "white"
                )
                textcolor = (
                    "white"
                    if i == image.get_meta("multimodule_index")
                    and multimodule_highlight_selection
                    else "black"
                )
                ax.text(
                    box[1].centroid.x,
                    box[1].centroid.y,
                    s=str(i),
                    color=textcolor,
                    fontsize=multimodule_numbers_fontsize,
                    bbox=dict(facecolor=bgcolor, alpha=0.8),
                    ha="center",
                    va="center",
                )

    def is_active(self, image: Image) -> bool:
        super().is_active(image)

        return image.has_meta("multimodule_boxes") and image.has_meta(
            "multimodule_index"
        )


def _register_default_plugins():
    register_show_plugin("show_cell_crossings", ShowCellCrossings())
    register_show_plugin("multimodule_results", ShowMultimoduleDetections())
