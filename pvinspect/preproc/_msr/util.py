from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch as t
from torch.nn import functional as F
from skimage import img_as_ubyte
from skimage.io import imsave
from matplotlib import pyplot as plt

from functools import wraps
import inspect


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names = list(inspect.signature(func).parameters)
    defaults = [x.default for x in inspect.signature(func).parameters.values()]

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def plot_pytorch_image(path: Path, data: t.Tensor, lognorm: bool = False) -> None:
    data = data.detach().cpu().numpy().transpose(1, 2, 0)
    fig, ax = plt.subplots(1, 1)

    if not lognorm:
        plot = ax.imshow(data)
        fig.colorbar(plot)
    else:
        norm = SymLogNorm(linthresh=0.01)
        plot = ax.matshow(data, norm=norm)
        fig.colorbar(plot)

    fig.savefig(path)
    plt.clf()
    plt.cla()


def save_pytorch_image(path: Path, data: t.Tensor) -> None:
    """Save a CHW PyTorch image

    Args:
        path (Path): Target file
        data (t.Tensor): PyTorch image in CHW-format
    """
    data = img_as_ubyte(
        t.clamp(data.detach(), 0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
    )
    imsave(path, data)


def show_pytorch_image(data: t.Tensor, ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 15))
    plot = ax.imshow(data.detach().cpu().numpy()[0], cmap="gray")


def show_pytorch_plot(data: t.Tensor, lognorm: bool = False, ax=None, fig=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    if not lognorm:
        plot = ax.imshow(data.detach().cpu().numpy()[0], cmap="inferno")
    else:
        norm = SymLogNorm(linthresh=0.01)
        plot = ax.matshow(data.detach().cpu().numpy()[0], norm=norm, cmap="inferno")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot, cax=cax, orientation="vertical")


def median_absolute_deviation(x: t.Tensor) -> t.Tensor:
    """Computes the median absolute deviation"""
    x = x.flatten()
    inner_med = t.median(x)
    outer_med = t.median(t.abs(x - inner_med))
    return outer_med


def median_filter(x: t.Tensor, size: int) -> t.Tensor:
    """Applies 2d median filtering

    Implementation inspired by https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    """
    x = F.pad(x, tuple([size // 2] * 4), mode="replicate")
    x = x.unfold(2, size, 1).unfold(3, size, 1)
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


def weighted_median(
    x: t.Tensor, w: t.Tensor, dim: int = -1, keep_dim: bool = False
) -> t.Tensor:
    # flatten
    if dim == -1:
        x = x.flatten()
        w = w.flatten()
        dim = 0

    # normalize weights
    w = w / w.sum(dim=dim, keepdim=True)

    # sort x and w according to x
    idx = t.argsort(x, dim=dim)
    x = t.gather(x, dim, idx)
    w = t.gather(w, dim, idx)

    # compute cumulative sum of weights
    wsum = t.cumsum(w, dim=dim)

    # first element > 0.5
    median_idx = t.nonzero(wsum > 0.5)[dim][0]
    x = x.transpose(0, dim)
    x = x[median_idx].unsqueeze(0)
    x = x.transpose(0, dim)

    if not keep_dim:
        x = x.squeeze(0)

    return x


def weighted_median_absolute_deviation(x: t.Tensor, w: t.Tensor) -> t.Tensor:
    x = x.flatten()
    w = w.flatten()
    inner_med = weighted_median(x, w)
    outer_med = weighted_median(t.abs(x - inner_med), w)
    return outer_med
