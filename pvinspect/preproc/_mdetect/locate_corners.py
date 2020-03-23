import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from skimage.transform.integral import integral_image
from scipy.ndimage.filters import gaussian_filter1d
from .config import (
    GAUSSIAN_RELATIVE_SIGMA,
    CORNER_DETECTION_MAX_PEAKS,
    CORNER_DETECTION_PEAK_THRESH,
)
from .summary import Summary
from scipy.signal import find_peaks

summary = Summary("locate_corners")


def inner_corners_model(n_cols, m_rows):
    x, y = np.meshgrid(np.arange(1, n_cols), np.arange(1, m_rows))
    return np.array([x.flatten(), y.flatten()], dtype=np.float64).T


def outer_corners_model(n_cols, m_rows):
    x = (
        list(range(0, n_cols + 1))
        + [n_cols] * (m_rows - 1)
        + list(reversed(range(0, n_cols + 1)))
        + [0] * (m_rows - 1)
    )
    y = (
        [0] * (n_cols + 1)
        + list(range(1, m_rows))
        + [m_rows] * (n_cols + 1)
        + list(reversed(range(1, m_rows)))
    )

    return np.array([x, y], dtype=np.float64).T


def _corner_patches(coords, patch_size):
    d = patch_size / 2
    coords = np.repeat(np.expand_dims(coords, 1), 4, 1)
    coords[:, 0] -= d
    coords[:, 1, 0] += d
    coords[:, 1, 1] -= d
    coords[:, 2] += d
    coords[:, 3, 0] -= d
    coords[:, 3, 1] += d
    return coords


def _extract_profile2d(img, patch, transform, n_samples):
    n_samples = 80
    x, y = np.meshgrid(
        np.linspace(patch[0, 0], patch[1, 0], n_samples),
        np.linspace(patch[0, 1], patch[3, 1], n_samples),
    )
    xy = np.array([x.flatten(), y.flatten()]).T
    xy_t = transform(xy)

    # simple projection to image bounds
    xy_t[xy_t[:, 0] < 0.0, 0] = 3
    xy_t[xy_t[:, 0] > img.shape[1] - 1, 0] = img.shape[1] - 3
    xy_t[xy_t[:, 1] < 0.0, 1] = 3  # 0 gives artifacts
    xy_t[xy_t[:, 1] > img.shape[0] - 1, 1] = img.shape[0] - 3

    # warp patch
    yx_t = np.flip(xy_t, 1)
    v = interpn(
        (np.arange(img.shape[0]), np.arange(img.shape[1])),
        img,
        yx_t,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    summary.put(
        "patch_{:d}_{:d}".format(int(patch[0, 0] + 1), int(patch[0, 1] + 1)),
        v.reshape(n_samples, n_samples),
    )
    return x, y, v.reshape(n_samples, n_samples)


def _find_stops(img, dim, sigma, patch, edgetype="ridge"):
    img = img.T if dim == 0 else img

    # calculate downsampling
    size = 0.5 * np.sum(img.shape)

    # extract profile of cumsum along di
    profile = np.sum(img, 1)
    profile_smooth = gaussian_filter1d(profile, sigma)
    profile_smooth = profile_smooth - np.min(profile_smooth)
    profile_smooth = profile_smooth / (np.max(profile_smooth) + 1e-5)

    # calculate gradient of that
    grad_smooth = np.gradient(profile_smooth)
    grad2_smooth = np.gradient(grad_smooth)

    summary.put(
        "patch_{:d}_{:d}_profile_{}".format(
            int(patch[0, 0] + 1), int(patch[0, 1] + 1), "x" if dim == 0 else "y"
        ),
        profile,
    )
    summary.put(
        "patch_{:d}_{:d}_profile_grad_{}".format(
            int(patch[0, 0] + 1), int(patch[0, 1] + 1), "x" if dim == 0 else "y"
        ),
        grad_smooth,
    )

    # find peaks in first and second derivative of profile
    min_distance = img.shape[0] // CORNER_DETECTION_MAX_PEAKS
    peaks, _ = find_peaks(
        grad2_smooth,
        distance=min_distance,
        height=CORNER_DETECTION_PEAK_THRESH * np.mean(np.abs(grad2_smooth)),
    )
    maxpeaks, _ = find_peaks(
        grad_smooth,
        distance=min_distance,
        height=CORNER_DETECTION_PEAK_THRESH * np.mean(np.abs(grad_smooth)),
    )
    minpeaks, _ = find_peaks(
        -grad_smooth,
        distance=min_distance,
        height=CORNER_DETECTION_PEAK_THRESH * np.mean(np.abs(grad_smooth)),
    )

    # plt.subplot(1,2,1)
    # plt.plot(np.arange(profile.shape[0]), grad_smooth)
    # plt.scatter(peaks, grad_smooth[peaks])
    # plt.subplot(1,2,2)
    # plt.plot(np.arange(profile.shape[0]), grad2_smooth)
    # plt.scatter(peaks, grad2_smooth[peaks])
    # plt.show()

    # threshold for finding minimum/maximum corresponding to peak in 2nd derivative
    eps = img.shape[0] // 10

    if edgetype in ("up", "down"):
        # filter all peaks in 2nd derivative that are not close to a maximum/minimum
        compare = maxpeaks if edgetype == "up" else minpeaks
        peaks_new = []
        for peak in peaks.tolist():
            if np.any(np.abs(peak - compare) <= eps):
                peaks_new.append(peak)

        # if multiple found, return the one closest to the center
        if len(peaks_new) > 1:
            center = img.shape[0] // 2
            distances = np.abs(np.array(peaks_new) - center)
            return peaks_new[np.argmin(distances)]
        elif len(peaks_new) == 1:
            return peaks_new[0]
        else:
            return None

    else:
        # filter all peaks in 2nd derivative that are not close to a maximum and a minimum
        peaks_new = []
        for peak in peaks.tolist():
            if np.any(np.abs(peak - maxpeaks) <= eps) and np.any(
                np.abs(peak - minpeaks) <= eps
            ):
                peaks_new.append(peak)

        # if number of remaining peaks uneven, return the middle one
        if len(peaks_new) % 2 == 1:
            return peaks_new[len(peaks_new) // 2]
        # otherwise take the one closest to the center
        elif len(peaks_new) > 0:
            center = img.shape[0] // 2
            distances = np.abs(np.array(peaks_new) - center)
            return peaks_new[np.argmin(distances)]
        else:
            return None


def _determine_sampling_and_filter(patch_t):
    patch_width = max(
        np.linalg.norm(patch_t[1] - patch_t[0]), np.linalg.norm(patch_t[3] - patch_t[2])
    )
    patch_height = max(
        np.linalg.norm(patch_t[2] - patch_t[1]), np.linalg.norm(patch_t[3] - patch_t[0])
    )
    n_samples = max(patch_height, patch_width)

    sigma = GAUSSIAN_RELATIVE_SIGMA * n_samples
    s = np.floor(2 / 3 * np.pi * sigma)

    if s > 1:
        sigma_s = (3 * s) / (2 * np.pi)
        if sigma ** 2 - sigma_s ** 2 < 0.5 ** 2:
            # convince Nyquist
            s -= 1
            sigma_s = (3 * s) / (2 * np.pi)
        sigma_filter = np.sqrt(sigma ** 2 - sigma_s ** 2)
        n_samples /= s
    else:
        sigma_filter = sigma

    return int(n_samples), sigma_filter


def fit_inner_corners(img, coords, transform, patch_size):
    patches = _corner_patches(coords, patch_size)
    accepted_flags = list()
    corners = list()

    # determine #samples and filter for one example only
    n_samples, sigma_filter = _determine_sampling_and_filter(
        transform(patches[len(patches) // 2])
    )

    for i in range(patches.shape[0]):
        _, _, v = _extract_profile2d(img, patches[i], transform, n_samples)

        # find stop in x and y direction
        xc = _find_stops(v, 0, sigma_filter, patches[i])
        yc = _find_stops(v, 1, sigma_filter, patches[i])

        # if both succeded
        if xc is not None and yc is not None:

            # xstops and ystops is given in patch coordinates -> transform to image coordinates
            x_size, y_size = v.shape[1], v.shape[0]
            corner = [xc / x_size, yc / y_size]
            accepted_flags.append(True)
            corners.append(corner)
        else:
            accepted_flags.append(False)
            corners.append([0, 0])

        # plt.imshow(v, cmap = plt.get_cmap('gray'))

        # if accepted:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'green')
        # else:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'red')

        # plt.show()
        # plt.clf()

    corners = np.array(corners)
    corners = corners - (patch_size / 2)
    return coords + corners, np.array(accepted_flags)


def fit_outer_corners(
    img, coords, transform, patch_size, n_cols, m_rows, is_module_detail
):

    if is_module_detail:
        # determine, on which sides the module is continued
        x_profile = np.sum(img, 1)
        y_profile = np.sum(img, 0)
        x_thresh = 0.3 * (np.max(x_profile) - np.min(x_profile)) + np.min(x_profile)
        y_thresh = 0.3 * (np.max(y_profile) - np.min(y_profile)) + np.min(y_profile)
        cont_left = np.median(y_profile[:10]) > y_thresh
        cont_right = np.median(y_profile[-10:]) > y_thresh
        cont_top = np.median(x_profile[:10]) > x_thresh
        cont_bottom = np.median(x_profile[-10:]) > x_thresh
        # plt.plot(np.arange(x_profile.shape[0]), x_profile)
        # plt.show()
        # plt.plot(np.arange(y_profile.shape[0]), y_profile)
        # plt.show()
        # print(x_thresh, y_thresh)
    else:
        cont_left, cont_right, cont_top, cont_bottom = False, False, False, False

    # depending on the orientation of the model, we need to assign differently
    x0 = transform(np.array([[0, 0]])).flatten()
    if np.linalg.norm(x0 - np.array([0, 0])) < np.linalg.norm(
        x0 - np.array([img.shape[0], 0])
    ):
        # horizontal orientation
        x_cont_left = cont_left
        x_cont_right = cont_right
        y_cont_left = cont_top
        y_cont_right = cont_bottom
    else:
        x_cont_left = cont_top
        x_cont_right = cont_bottom
        y_cont_left = cont_right
        y_cont_right = cont_left

    patches = _corner_patches(coords, patch_size)
    accepted_flags = list()
    corners = list()

    # determine #samples and filter for one example only
    n_samples, sigma_filter = _determine_sampling_and_filter(
        transform(patches[len(patches) // 2])
    )

    for i in range(patches.shape[0]):
        _, _, v = _extract_profile2d(img, patches[i], transform, n_samples)
        # plt.imshow(v, cmap='gray')
        # plt.show()

        if i <= n_cols and not y_cont_left:
            y_edgetype = "up"
        elif i >= n_cols + m_rows and i <= 2 * n_cols + m_rows and not y_cont_right:
            y_edgetype = "down"
        else:
            y_edgetype = "ridge"

        if i >= n_cols and i <= n_cols + m_rows and not x_cont_right:
            x_edgetype = "down"
        elif (i >= 2 * n_cols + m_rows or i == 0) and not x_cont_left:
            x_edgetype = "up"
        else:
            x_edgetype = "ridge"

        xc = _find_stops(v, 0, sigma_filter, patches[i], x_edgetype)
        yc = _find_stops(v, 1, sigma_filter, patches[i], y_edgetype)

        if xc is not None and yc is not None:
            x_size, y_size = v.shape[1], v.shape[0]
            xc /= x_size
            yc /= y_size
            corner = [xc, yc]
            accepted_flags.append(True)
            corners.append(corner)
        else:
            accepted_flags.append(False)
            corners.append([0, 0])

        # plt.imshow(v, cmap = plt.get_cmap('gray'))

        # if accepted:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'green')
        # else:
        #    plt.plot([corner[0]*x_size], [corner[1]*y_size], marker = 'x', color = 'red')

        # plt.show()
        # plt.clf()

    corners = np.array(corners)
    corners -= patch_size / 2
    return coords + corners, np.array(accepted_flags)
