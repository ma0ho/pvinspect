from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import numpy as np

from pvinspect.data.image import Image
from pvinspect.data.image.show_plugin import ShowPlugin, register_show_plugin


class ShowImage(ShowPlugin):
    def __init__(self, title: str = "image options", priority: int = -100):
        super().__init__(title, priority=priority)

    def apply(
        self,
        ax: Axes,
        image: Image,
        clip_low: float = 0.001,
        clip_high: float = 99.999,
        colorbar: bool = True,
        **kwargs
    ) -> None:
        """This plugin draws the image

        Args:
            clip_low (float): Clip values below this percentile
            clip_high (float): Clip values above this percentile
            colorbar (bool): Show the colorbar
            return super().apply(ax, image, **kwargs)
        """
        super().apply(ax, image, **kwargs)
        clip_low = clip_low if clip_low is not None else 0.0
        clip_high = clip_high if clip_high is not None else 100.0
        p = np.percentile(image.data, q=[clip_low, clip_high])  # type: ignore    # TODO: Check why this gives type error
        d = np.clip(image.data, p[0], p[1])
        ax.imshow(d, cmap="gray")
        if colorbar:
            plt.colorbar(ax=ax)

    def is_active(self, image: Image) -> bool:
        super().is_active(image)
        return True


class AxisOptions(ShowPlugin):
    def __init__(self, title: str = "axis options", priority: int = -200):
        super().__init__(title, priority=priority)

    def apply(
        self,
        ax: Axes,
        image: Image,
        show_axis: bool = True,
        show_title: bool = True,
        max_title_length: int = 30,
        **kwargs
    ) -> None:
        """Draw the axes

        Args:
            show_axis (bool): Show the axis
            show_title (bool): Show the image title
            max_title_length (int): Maximum length of image title
        """
        super().apply(ax, image, **kwargs)
        if not show_axis:
            ax.set_axis_off()
        if show_title:
            if image.path is not None:
                t = str(image.path.name)
            else:
                t = ""

            if len(t) > max_title_length:
                l1 = max_title_length // 2 - 2
                l2 = max_title_length - l1 - 2
                t = t[:l1] + ".." + t[len(t) - l2 :]

            ax.set_title(t)

    def is_active(self, image: Image) -> bool:
        super().is_active(image)
        return True


def _register_default_plugins():
    register_show_plugin("show_image", ShowImage())
    register_show_plugin("axis_options", AxisOptions())
