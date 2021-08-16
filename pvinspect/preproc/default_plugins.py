import numpy as np
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


def _register_default_plugins():
    register_show_plugin("show_cell_crossings", ShowCellCrossings())
