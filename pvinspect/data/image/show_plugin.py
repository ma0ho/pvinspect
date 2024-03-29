from __future__ import annotations

from os import write

import matplotlib

"""Plugin mechanism for image viewer"""

import copy
import inspect
import logging
import math
import re
import sys
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from docstring_parser import parse
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from skimage import img_as_float64, img_as_int, img_as_uint
from tqdm import tqdm


class PluginOption(NamedTuple):
    name: str
    type_name: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None
    default: Any = None


class ShowPlugin(metaclass=ABCMeta):
    def __init__(self, title: str, priority: int = 0):
        """Base show plugin

        Args:
            title (str): Title of this plugin
            priority (int): Plugins are invoked in the order of increasing priority (highest priority is invoked
                last and hence appears on top)
        """
        self._title = title
        self._priority = priority

    @property
    def title(self):
        return self._title

    @property
    def priority(self):
        return self._priority

    @abstractmethod
    def apply(self, ax: Axes, image, **kwargs) -> None:
        pass

    @abstractmethod
    def is_active(self, image) -> bool:
        pass

    @classmethod
    def options(cls) -> List[PluginOption]:
        res: List[PluginOption] = list()

        # TODO: Determine default arguments from getfullargspec
        if cls.apply.__doc__:
            doc = parse(cls.apply.__doc__)
            for p in doc.params:
                if p.arg_name not in ("ax", "image"):
                    res.append(
                        PluginOption(
                            name=p.arg_name,
                            type_name=p.type_name,
                            description=p.description,
                            optional=p.is_optional,
                        )
                    )

        return res


# this is a pointer to the module object instance itself
this: Any = sys.modules[__name__]

# global list of plugins that are called on every .show()
this.show_plugins = dict()
this.show_plugins_sorted = None


def _build_cache():
    # conditionally build cache
    if this.show_plugins_sorted is None:
        this.show_plugins_sorted = sorted(
            this.show_plugins.values(), key=lambda x: x.priority
        )


def register_show_plugin(name: str, plugin: ShowPlugin):
    """Register a new plugin that is called on every .show()

    Args:
        name (str): Name referencing this particular instance
        plugin (ShowPlugin):
    """
    this.show_plugins[name] = plugin
    # print(this.show_plugins)

    # reset cache
    this.show_plugins_sorted = None


def get_active_show_plugins(image) -> List[ShowPlugin]:
    """Get active show plugins ordered by priority"""
    _build_cache()
    return [p for p in this.show_plugins_sorted if p.is_active(image)]


def invoke_show_plugins(image, ax: Axes, **kwargs):
    """Run the stack of registered and active plugins on given image

    Args:
        image (Image)
        ax (Axes): The axes object that will be used for plotting
    """
    _build_cache()
    # print(this.show_plugins)
    for plugin in this.show_plugins_sorted:
        if plugin.is_active(image):
            plugin.apply(ax, image, **kwargs)


# def _register_default_plugins():
#    def show_cell_crossings(
#        image: Image, show_cell_crossings: bool = True, **kwargs
#    ):
#        if (
#            show_cell_crossings
#            and isinstance(image, Image)
#            and image.has_meta("transform")
#        ):
#            grid = image.grid()
#            coords = image.get_meta("transform").__call__(grid)
#            plt.scatter(coords[:, 0], coords[:, 1], c="yellow", marker="+")
#
#    register_show_plugin(show_cell_crossings)
#
#    def multimodule_show_boxes(
#        image: Image,
#        multimodule_show_boxes: bool = True,
#        multimodule_highlight_selection: bool = True,
#        multimodule_boxes_linewidth: int = 2,
#        **kwargs
#    ):
#        if (
#            multimodule_show_boxes
#            and isinstance(image, Image)
#            and image.has_meta("multimodule_boxes")
#        ):
#            for i, box in enumerate(image.get_meta("multimodule_boxes")):
#                color = (
#                    "red"
#                    if i == image.get_meta("multimodule_index")
#                    and multimodule_highlight_selection
#                    else "yellow"
#                )
#                plt.plot(
#                    *box[1].exterior.xy,
#                    linewidth=multimodule_boxes_linewidth,
#                    color=color,
#                )
#
#    register_show_plugin(multimodule_show_boxes)
#
#    def multimodule_show_numbers(
#        image: Image,
#        multimodule_show_numbers: bool = True,
#        multimodule_highlight_selection: bool = True,
#        multimodule_numbers_fontsize: int = 20,
#        **kwargs
#    ):
#        if (
#            multimodule_show_numbers
#            and isinstance(image, Image)
#            and image.has_meta("multimodule_boxes")
#        ):
#            for i, box in enumerate(image.get_meta("multimodule_boxes")):
#                bgcolor = (
#                    "red"
#                    if i == image.get_meta("multimodule_index")
#                    and multimodule_highlight_selection
#                    else "white"
#                )
#                textcolor = (
#                    "white"
#                    if i == image.get_meta("multimodule_index")
#                    and multimodule_highlight_selection
#                    else "black"
#                )
#                plt.text(
#                    box[1].centroid.x,
#                    box[1].centroid.y,
#                    s=str(i),
#                    color=textcolor,
#                    fontsize=multimodule_numbers_fontsize,
#                    bbox=dict(facecolor=bgcolor, alpha=0.8),
#                    ha="center",
#                    va="center",
#                )
#
#    register_show_plugin(multimodule_show_numbers)
#
#    def calibration_show_reference_box(
#        image: Image,
#        calibration_show_reference_box: bool = True,
#        calibration_reference_box_color="red",
#        **kwargs
#    ):
#        if (
#            calibration_show_reference_box
#            and isinstance(image, Image)
#            and image.has_meta("calibration_reference_box")
#        ):
#            plt.plot(
#                *image.get_meta("calibration_reference_box").exterior.xy,
#                # linewidth=multimodule_boxes_linewidth,
#                color=calibration_reference_box_color,
#            )
#
#    register_show_plugin(calibration_show_reference_box)
#
#    def segment_module_show_box(
#        image: Image,
#        segment_module_show_box: bool = True,
#        segment_module_show_box_color="red",
#        **kwargs
#    ):
#        if (
#            segment_module_show_box
#            and isinstance(image, Image)
#            and image.has_meta("segment_module_original_box")
#        ):
#            plt.plot(
#                *image.get_meta("segment_module_original_box").exterior.xy,
#                color=segment_module_show_box_color,
#            )
#
#    register_show_plugin(segment_module_show_box)
#
