import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq as optimize_least_squares
from scipy.optimize import minimize
from scipy.ndimage import sobel


class OptimimzerStitching:
    def __init__(
        self,
        images,
        points,
        sigmas,
        window_size,
        use_gradients=True,
        binarize_grad=True,
    ):
        self.images = images.copy()
        self.points = points.copy()

    def stitch(self):

        # upper left and right
        img_1 = self.images[0].astype(np.float64)
        img_2 = self.images[1].astype(np.float64)
        points_1 = self.points[0]
        points_2 = self.points[1]

        # subtract local mean
        f_1 = medfilt2d(img_1, 31)
        f_2 = medfilt2d(img_2, 31)
        img_1 -= f_1
        img_2 -= f_2

        # get rectification transforms
        target_1 = np.array(
            [(0.0, 0.0), (3.0, 0.0), (0.0, 5.0), (3.0, 5.0)]
        )  # cells have unit length
        target_2 = target_1 + np.array([(3.0, 0.0)])
        H_1, _ = cv2.findHomography(points_1, target_1)
        H_2, _ = cv2.findHomography(points_2, target_2)

        # fix resolution of target
        cell_size = (points_1[1, 0] - points_1[0, 0]) / 3
        S = np.diag([cell_size, cell_size, 1])  # scale matrix

        # Test: Initially warp img_1 and img_2
        # res = self._warp_both(img_1, img_2, H_1, H_2, S, (6,5))
        # plt.imshow(res)
        # plt.show()

        # find ROI in target coordinate system
        lb = np.array([(0.0, 0.0), (0.0, img_2.shape[0])], dtype=np.float64).reshape(
            -1, 1, 2
        )
        rb = np.array(
            [(img_1.shape[1], 0.0), (img_1.shape[1], img_1.shape[0])], dtype=np.float64
        ).reshape(-1, 1, 2)
        img_2_left_border = cv2.perspectiveTransform(lb, H_2).reshape(-1, 2)
        img_1_right_border = cv2.perspectiveTransform(rb, H_1).reshape(-1, 2)
        rect_x1 = max(img_2_left_border[0, 0], img_2_left_border[1, 0])
        rect_y1 = max(img_2_left_border[0, 1], img_1_right_border[0, 1])
        rect_x2 = min(img_1_right_border[0, 0], img_1_right_border[1, 0])
        rect_y2 = min(img_1_right_border[1, 1], img_2_left_border[1, 1])
        roi = np.array(
            [
                (rect_x1, rect_y1),
                (rect_x2, rect_y1),
                (rect_x1, rect_y2),
                (rect_x2, rect_y2),
            ]
        )
        centroid = np.mean(roi, axis=0, keepdims=True)
        print(roi)
        print(centroid)
        roi = (
            roi - centroid
        ) * 0.75 + centroid  # only use inner 75% of ROI so that we can keep that constant during optimization
        roi_w = roi[1, 0] - roi[0, 0]
        roi_h = roi[2, 1] - roi[0, 1]
        roi_area = (roi_w, roi_h)

        # fix H_1 and H_2 so that the top left edge of roi maps to (0, 0)
        fix = np.diag([1.0, 1.0, 1.0])
        fix[:2, 2] = -roi[0]
        H_1_roi = fix.dot(H_1)
        H_2_roi = fix.dot(H_2)

        # Test: plot roi in both images
        self._plot_roi(img_1, img_2, H_1_roi, H_2_roi, S, roi_area)

        # iterate over scales, where s is the amount of downscaling,
        # eg. s=8 means that images are downscaled by a factor of 8
        for s in (8, 4, 2, 1):
            S_scaled = np.diag([1 / s, 1 / s, 1]).dot(S)
            sigma = (
                3 * s / (2 * np.pi)
            )  # standard deviation of filter that cuts at Nyquist frequency according to s
            img_1_f = gaussian_filter(img_1, sigma)
            img_2_f = gaussian_filter(img_2, sigma)

            # initial parameters:
            # parameters give pertubation of 4 corner points of the ROI in the second image
            x_init = np.zeros(8, dtype=np.float64)

            # we keep H_1 fixed, so we can precompute roi_1
            roi_1 = self._warp(img_1_f, H_1_roi, S_scaled, roi_area)

            def opt_fn(x):
                H_corr, _ = cv2.findHomography(roi, roi + x.reshape(4, 2))
                roi_2 = self._warp(img_2_f, H_corr.dot(H_2_roi), S_scaled, roi_area)
                return -np.mean(roi_1 * roi_2)

            # run optimization
            res = minimize(opt_fn, x_init, method="COBYLA", options={"tol": 1e-12})
            print(res)

            # Test: plot final state
            H_corr, _ = cv2.findHomography(roi, roi + res.x.reshape(4, 2))
            if s == 1:
                self._plot_roi(
                    img_1_f, img_2_f, H_1_roi, H_2_roi, S_scaled, roi_area, H_corr
                )

            # update H_2_roi
            H_2_roi = H_corr.dot(H_2_roi)

        # undo shift of H_2
        H_2_old = H_2.copy()
        fix[:2, 2] = roi[0]
        H_2 = fix.dot(H_2_roi)

        result_old = self._warp_both(img_1, img_2, H_1, H_2_old, S, (6, 5))
        result = self._warp_both(img_1, img_2, H_1, H_2, S, (6, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(result_old)
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.show()

        return None

    def _plot_roi(self, img_1, img_2, H_1_roi, H_2_roi, S, area, H_corr=None):
        roi_1 = self._warp(img_1, H_1_roi, S, area)
        roi_2 = self._warp(img_2, H_2_roi, S, area)

        if H_corr is None:
            plt.subplot(1, 4, 1)
            plt.imshow(roi_1)
            plt.subplot(1, 4, 2)
            plt.imshow(roi_2)
            plt.subplot(1, 4, 3)
            plt.imshow(-(roi_1 * roi_2))
            plt.subplot(1, 4, 4)
            plt.imshow(roi_1 - roi_2)
        else:
            roi_res = self._warp(img_2, H_corr.dot(H_2_roi), S, area)
            plt.subplot(2, 4, 1)
            plt.imshow(roi_1)
            plt.subplot(2, 4, 2)
            plt.imshow(roi_2)
            plt.subplot(2, 4, 3)
            plt.imshow(-(roi_1 * roi_2))
            plt.subplot(2, 4, 4)
            plt.imshow(roi_1 - roi_2)

            plt.subplot(2, 4, 5)
            plt.imshow(roi_1)
            plt.subplot(2, 4, 6)
            plt.imshow(roi_res)
            plt.subplot(2, 4, 7)
            plt.imshow(-(roi_1 * roi_res))
            plt.subplot(2, 4, 8)
            plt.imshow(roi_1 - roi_res)

        plt.show()

    def _warp(self, img, H, S, area):
        size = (int(area[0] * S[0, 0]), int(area[1] * S[1, 1]))
        return cv2.warpPerspective(img, S.dot(H), size)

    def _warp_both(self, img_1, img_2, H_1, H_2, S, area):
        size = (int(area[0] * S[0, 0]), int(area[1] * S[1, 1]))
        res1 = self._warp(img_1, H_1, S, area)
        res1[:, size[0] // 2 :] = 0.0
        res2 = self._warp(img_2, H_2, S, area)
        res2[:, : size[0] // 2] = 0.0
        return res1 + res2
