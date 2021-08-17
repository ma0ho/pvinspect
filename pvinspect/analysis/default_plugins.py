import numpy as np
from matplotlib.axes import Axes
from pvinspect.data.image import Image
from pvinspect.data.image.show_plugin import ShowPlugin, register_show_plugin
from skimage.transform import resize


class ShowCrackCAM(ShowPlugin):
    def __init__(self, title: str = "crack CAM overlay", priority: int = -50):
        super().__init__(title, priority=priority)

    def apply(
        self,
        ax: Axes,
        image: Image,
        show_cracks: bool = True,
        crack_alpha: float = 0.5,
        **kwargs
    ) -> None:
        """Draw the axes

        Args:
            show_cracks (bool): Show an overlay indicating the crack positions
            crack_alpha (float): Transparency (0..1) of crack overlay
        """
        super().apply(ax, image, **kwargs)

        if show_cracks:
            resized_cam = (
                resize(image.get_meta("pred_crack_cam"), image.shape, order=1) > 0.5
            )
            cam_rgba = np.zeros((resized_cam.shape[0], resized_cam.shape[1], 4))
            cam_rgba[:, :, 0] = 1
            cam_rgba[:, :, 3] = resized_cam * crack_alpha
            ax.imshow(cam_rgba)

    def is_active(self, image: Image) -> bool:
        super().is_active(image)
        return image.has_meta("pred_crack_cam")


def _register_default_plugins():
    register_show_plugin("show_crack_cam", ShowCrackCAM())
