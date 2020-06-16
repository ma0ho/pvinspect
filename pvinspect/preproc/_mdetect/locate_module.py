import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter1d
import math
from .config import (
    GAUSSIAN_RELATIVE_SIGMA,
    OUTER_CORNER_THRESH_FACTOR,
    MODULE_DETECTION_PEAK_THRESH,
)
from scipy import optimize, signal
from pvinspect.common import transform
from .summary import Summary

summary = Summary("locate_module")


def _find_stops(img, dim, n_cells):
    img = img.T if dim == 0 else img

    # calculate downsampling
    size = 0.5 * np.sum(img.shape)

    # extract profile of cumsum along di
    profile = np.sum(img, 1)
    profile_smooth = gaussian_filter1d(profile, GAUSSIAN_RELATIVE_SIGMA * size)
    profile_smooth = profile_smooth - np.min(profile_smooth)
    profile_smooth = profile_smooth / (np.max(profile_smooth) + 1e-5)

    # calculate gradient of that
    grad_smooth = np.gradient(profile_smooth)

    thresh = MODULE_DETECTION_PEAK_THRESH * np.mean(np.abs(grad_smooth))
    peaks_max, _ = signal.find_peaks(grad_smooth, height=thresh)
    peaks_min, _ = signal.find_peaks(-grad_smooth, height=thresh)

    if len(peaks_max) == 0 or len(peaks_min) == 0:
        return None

    extremals = [peaks_max[0], peaks_min[-1]]

    thresh = np.std(grad_smooth) * OUTER_CORNER_THRESH_FACTOR

    min_distance = int(
        (extremals[1] - extremals[0]) / n_cells / 2
    )  # consider std only for half cell size

    res = []
    thresh = (
        np.std(
            np.clip(grad_smooth, 0.0, None)[
                max(0, extremals[0] - min_distance) : min(
                    img.shape[0], extremals[0] + min_distance
                )
            ]
        )
        * OUTER_CORNER_THRESH_FACTOR
    )
    res.append(extremals[0] - np.argmax((grad_smooth <= thresh)[extremals[0] :: -1]))
    res.append(extremals[0] + np.argmax((grad_smooth <= thresh)[extremals[0] :: +1]))
    thresh = (
        np.std(
            np.clip(grad_smooth, None, 0.0)[
                max(0, extremals[1] - min_distance) : min(
                    img.shape[0], extremals[1] + min_distance
                )
            ]
        )
        * OUTER_CORNER_THRESH_FACTOR
    )
    res.append(extremals[1] - np.argmax((grad_smooth >= -thresh)[extremals[1] :: -1]))
    res.append(extremals[1] + np.argmax((grad_smooth >= -thresh)[extremals[1] :: +1]))

    # plt.plot(np.arange(profile.shape[0]), grad_smooth)
    # plt.scatter(peaks_max, grad_smooth[peaks_max])
    # plt.scatter(peaks_min, grad_smooth[peaks_min])
    # plt.scatter(res, [grad_smooth[x] for x in res])
    # plt.show()

    return res


def _assign_stops(x_stops, y_stops, img):

    # find outer and inner bounding box
    outer_anchor = (min(x_stops[0], x_stops[1]), min(y_stops[0], y_stops[1]))
    inner_anchor = (max(x_stops[0], x_stops[1]), max(y_stops[0], y_stops[1]))
    outer_size = (
        max(x_stops[2], x_stops[3]) - outer_anchor[0] + 1,
        max(y_stops[2], y_stops[3]) - outer_anchor[1] + 1,
    )
    inner_size = (
        min(x_stops[2], x_stops[3]) - inner_anchor[0] + 1,
        min(y_stops[2], y_stops[3]) - inner_anchor[1] + 1,
    )
    # fig, ax = plt.subplots(1)
    # ax.imshow(img, cmap='gray')
    # rect_outer = patches.Rectangle(outer_anchor, *outer_size, linewidth=1, edgecolor='r', facecolor='none')
    # rect_inner = patches.Rectangle(inner_anchor, *inner_size, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect_outer)
    # ax.add_patch(rect_inner)
    # plt.show()

    # find boxes between outer and inner
    upper_margin = (
        max(x_stops[0], x_stops[1]),
        min(y_stops[0], y_stops[1]),
        min(x_stops[2], x_stops[3]),
        max(y_stops[0], y_stops[1]),
    )
    left_margin = (
        min(x_stops[0], x_stops[1]),
        max(y_stops[0], y_stops[1]),
        max(x_stops[0], x_stops[1]),
        min(y_stops[2], y_stops[3]),
    )
    bottom_margin = (
        max(x_stops[0], x_stops[1]),
        min(y_stops[2], y_stops[3]),
        min(x_stops[2], x_stops[3]),
        max(y_stops[2], y_stops[3]),
    )
    right_margin = (
        min(x_stops[2], x_stops[3]),
        max(y_stops[0], y_stops[1]),
        max(x_stops[2], x_stops[3]),
        min(y_stops[2], y_stops[3]),
    )

    # divide them into left/right or top/bottom part
    upper_margin_l = (
        upper_margin[1],
        upper_margin[0],
        upper_margin[3],
        upper_margin[0] + (upper_margin[2] - upper_margin[0] + 1) // 2,
    )
    upper_margin_r = (
        upper_margin[1],
        upper_margin_l[3],
        upper_margin[3],
        upper_margin[2],
    )
    left_margin_t = (
        left_margin[1],
        left_margin[0],
        left_margin[1] + (left_margin[3] - left_margin[1] + 1) // 2,
        left_margin[2],
    )
    left_margin_b = (left_margin_t[2], left_margin_t[1], left_margin[3], left_margin[2])
    bottom_margin_l = (
        bottom_margin[1],
        bottom_margin[0],
        bottom_margin[3],
        bottom_margin[0] + (bottom_margin[2] - bottom_margin[0] + 1) // 2,
    )
    bottom_margin_r = (
        bottom_margin[1],
        bottom_margin_l[3],
        bottom_margin[3],
        bottom_margin[2],
    )
    right_margin_t = (
        right_margin[1],
        right_margin[0],
        right_margin[1] + (right_margin[3] - right_margin[1] + 1) // 2,
        right_margin[2],
    )
    right_margin_b = (
        right_margin_t[2],
        right_margin[0],
        right_margin[3],
        right_margin[2],
    )

    summary.put("box_upper_l", upper_margin_l)
    summary.put("box_upper_r", upper_margin_r)
    summary.put("box_left_t", left_margin_t)
    summary.put("box_left_b", left_margin_b)
    summary.put("box_bottom_l", bottom_margin_l)
    summary.put("box_bottom_r", bottom_margin_r)
    summary.put("box_right_t", right_margin_t)
    summary.put("box_right_b", right_margin_b)

    def sum_helper(img, p0, p1):
        s0 = int(np.ceil((p1[0] - p0[0]) / 10))  # sample approx. 10 lines per dim
        s1 = int(np.ceil((p1[1] - p0[1]) / 10))
        s0 = s0 if s0 > 0 else 1
        s1 = s1 if s1 > 0 else 1
        return np.sum(img[p0[0] : p1[0] + 1 : s0, p0[1] : p1[1] + 1 : s1])

    # sum over parts
    sum_upper_margin_l = sum_helper(img, upper_margin_l[:2], upper_margin_l[2:])
    sum_upper_margin_r = sum_helper(img, upper_margin_r[:2], upper_margin_r[2:])
    sum_left_margin_t = sum_helper(img, left_margin_t[:2], left_margin_t[2:])
    sum_left_margin_b = sum_helper(img, left_margin_b[:2], left_margin_b[2:])
    sum_bottom_margin_l = sum_helper(img, bottom_margin_l[:2], bottom_margin_l[2:])
    sum_bottom_margin_r = sum_helper(img, bottom_margin_r[:2], bottom_margin_r[2:])
    sum_right_margin_t = sum_helper(img, right_margin_t[:2], right_margin_t[2:])
    sum_right_margin_b = sum_helper(img, right_margin_b[:2], right_margin_b[2:])

    # assign stops
    if sum_upper_margin_l > sum_upper_margin_r:
        Ay = y_stops[0]
        By = y_stops[1]
    else:
        Ay = y_stops[1]
        By = y_stops[0]

    if sum_bottom_margin_l > sum_bottom_margin_r:
        Cy = y_stops[2]
        Dy = y_stops[3]
    else:
        Cy = y_stops[3]
        Dy = y_stops[2]

    if sum_left_margin_t > sum_left_margin_b:
        Ax = x_stops[0]
        Dx = x_stops[1]
    else:
        Ax = x_stops[1]
        Dx = x_stops[0]

    if sum_right_margin_t > sum_right_margin_b:
        Bx = x_stops[3]
        Cx = x_stops[2]
    else:
        Bx = x_stops[2]
        Cx = x_stops[3]

    return np.array([(Ax, Ay), (Bx, By), (Cx, Cy), (Dx, Dy)])


def locate_module(img, n_cols, m_rows):
    x_stops = _find_stops(img, 0, n_cols)
    y_stops = _find_stops(img, 1, m_rows)

    if x_stops is None or y_stops is None:
        return None
    else:
        return _assign_stops(x_stops, y_stops, img)


def module_boundingbox_model(coords, n_cols, m_rows, orientation):
    if orientation is None:
        mean_x = 1 / 2 * (coords[1, 0] - coords[0, 0] + coords[2, 0] - coords[3, 0])
        mean_y = 1 / 2 * (coords[2, 1] - coords[1, 1] + coords[3, 1] - coords[0, 1])
        oriented_horizontal = mean_x > mean_y
    else:
        oriented_horizontal = orientation == "horizontal"

    if oriented_horizontal:
        src = np.array([[0, 0], [n_cols, 0], [n_cols, m_rows], [0, m_rows]])
    else:
        src = np.array([[0, m_rows], [0, 0], [n_cols, 0], [n_cols, m_rows]])

    return src
