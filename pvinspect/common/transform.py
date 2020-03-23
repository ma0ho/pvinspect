import numpy as np
from copy import copy, deepcopy
import scipy.optimize
import scipy.interpolate
from abc import ABC, abstractmethod, abstractproperty
from functools import reduce
import cv2
from . import util


def _brown(coords, coeffs):
    N = len(coeffs)

    radius = np.sqrt(np.sum(coords ** 2, axis=1))
    factor = 1 + np.expand_dims(
        np.sum([kn * (radius ** (2 * (n + 1))) for n, kn in enumerate(coeffs)], axis=0),
        axis=-1,
    )
    coords = coords * factor

    return coords


def _brown_oder1_inverse(coords, k):

    if k == 0.0:
        return coords

    ru = np.sqrt(np.sum(coords ** 2, axis=1))
    D = (27 * k ** 2 * ru) ** 2 + 4 * (3 * k) ** 3

    rd = np.empty_like(D)
    g1 = D >= 0
    if np.any(g1):
        tmp = np.sqrt((ru[g1] ** 2) / 4 + 1 / (27 * k))
        rd[g1] = np.cbrt(1 / k * (ru[g1] / 2 + tmp)) + np.cbrt(
            1 / k * (ru[g1] / 2 - tmp)
        )
    g2 = D < 0
    if np.any(g2):
        rd[g2] = (
            2
            * np.sqrt(-3 * k)
            / (3 * k)
            * np.cos(
                1 / 3 * np.arccos(27 * k ** 2 * ru[g2] / (2 * np.sqrt(-((3 * k) ** 3))))
                + 4 * np.pi / 3
            )
        )

    factor = rd / ru
    coords = coords * factor.reshape(-1, 1)

    return coords


def distort(coords, params):
    return _brown(coords, params)


def distort_inverse(coords, params):
    if len(params) > 1:
        raise NotImplementedError(
            "Inversion not implemented for more than 1 coefficient"
        )
    return _brown_oder1_inverse(coords, params[0])


def _normalize_transform(coords, new_centroid=(0, 0), new_mean_distance=np.sqrt(2)):
    centroid = np.mean(coords, axis=0)
    scaling = new_mean_distance / np.mean(
        np.linalg.norm(coords - centroid.reshape(1, -1), axis=1)
    )
    return np.array([[scaling, 0, -centroid[0]], [0, scaling, -centroid[1]], [0, 0, 1]])


def find_homography(src, dest, compute_residual=False):
    """
        Estimate homography using normalized DLT

        TODO How handle cases where H[2,2] == 0?
    """
    N = src.shape[0]
    dt = src.dtype
    normalize = N > 4

    if normalize:
        # compute transforms that normalize coordinates such that
        # centroid is at origin with average distance of sqrt(2)
        src_T = _normalize_transform(src)
        dest_T = _normalize_transform(dest)

        # normalize
        src_n = apply_homography(src, src_T)
        dest_n = apply_homography(dest, dest_T)
    else:
        src_n = src
        dest_n = dest

    # DLT
    A = np.zeros((2 * N, 9), dtype=np.float64)
    A[:N, 0] = src_n[:, 0]
    A[:N, 1] = src_n[:, 1]
    A[:N, 2] = 1
    A[:N, 6] = -dest_n[:, 0] * src_n[:, 0]
    A[:N, 7] = -dest_n[:, 0] * src_n[:, 1]
    A[:N, 8] = -dest_n[:, 0]
    A[N:, 3] = -src_n[:, 0]
    A[N:, 4] = -src_n[:, 1]
    A[N:, 5] = -1
    A[N:, 6] = dest_n[:, 1] * src_n[:, 0]
    A[N:, 7] = dest_n[:, 1] * src_n[:, 1]
    A[N:, 8] = dest_n[:, 1]

    U, sigma, VT = np.linalg.svd(A)
    # residual = sigma[-1]
    H = VT[-1].reshape(3, 3)

    # denormalize
    if normalize:
        H = np.linalg.inv(dest_T).dot(H).dot(src_T)

    if np.abs(H[2, 2]) != 0.0:
        H /= H[2, 2]
    else:
        H = None

    if compute_residual and normalize:
        if H is not None:
            tmp = apply_homography(src, H)
            residual = np.mean(np.linalg.norm(tmp - dest, ord=2, axis=1))
            return H, residual
        else:
            return None, None
    elif compute_residual:
        return H, sigma[-1]
    else:
        return H


def apply_homography(src, H):
    assert src.shape[-1] == 2
    src_flat = src.reshape(-1, 2)
    res = H[:, :2].dot(src_flat.T)
    res += H[:, 2].reshape(3, 1)
    with np.errstate(divide="ignore"):
        res[:2] /= res[2]
    return res[:2].T.reshape(src.shape)


def apply_pinhole(src, Rt, A, d, inverse=False):
    if not inverse:
        Rt = Rt[:, [0, 1, 3]]
        src = apply_homography(src, Rt)
        src = distort(src, d)
        return apply_homography(src, A)
    else:
        Rt = np.linalg.pinv(Rt[:, [0, 1, 3]])
        A = np.linalg.pinv(A)
        src = apply_homography(src, A)
        src = distort_inverse(src, d)
        return apply_homography(src, Rt)


def find_homography_ransac(src, dest, error_thresh_px, compute_residual=False):
    def model_f(mask):
        return find_homography(src[mask], dest[mask])

    def err_f(model):
        src_t = apply_homography(src, model)
        return np.linalg.norm(src_t - dest, axis=1) < error_thresh_px

    return util.ransac(src.shape[0], 4, model_f, err_f)


def _solve_PnP_wrap(src, dest, A, dist, n_dist_coeff):
    # coords in OpenCV compatible format
    src2 = []
    for src_i in src:
        src_i = np.concatenate([src_i, np.zeros((src_i.shape[0], 1))], axis=1)
        src2.append(src_i.astype(np.float32))
    dest2 = [x.astype(np.float32) for x in dest]

    Rt = list()
    if n_dist_coeff <= 4:
        k = np.zeros(4)
    elif n_dist_coeff <= 5:
        k = np.zeros(5)
    elif n_dist_coeff <= 8:
        k = np.zeros(8)
    k[:n_dist_coeff] = dist

    for s, d in zip(src2, dest2):
        retval, rvec, tvec = cv2.solvePnP(s, d, A, k)
        R = util.rodrigues2matrix(rvec)
        Rt.append(np.concatenate([R, tvec.reshape(3, 1)], axis=1))

    return Rt


def _calibrate_camera_wrap(
    src, dest, image_width, image_height, n_dist_coeff, mask=None
):
    # coords in OpenCV compatible format
    src2 = []
    for src_i in src:
        src_i = np.concatenate([src_i, np.zeros((src_i.shape[0], 1))], axis=1)
        src2.append(src_i.astype(np.float32))
    dest2 = [x.astype(np.float32) for x in dest]

    # mask
    if mask is not None:
        src2_m, dest2_m = util.filter_lists([src2, dest2], mask)
    else:
        src2_m, dest2_m = src2, dest2

    # calibrate
    if n_dist_coeff > 4 or n_dist_coeff == 3:
        raise NotImplementedError()
    else:
        flags = cv2.CALIB_FIX_K3
        if n_dist_coeff < 3:
            flags += cv2.CALIB_ZERO_TANGENT_DIST
        if n_dist_coeff < 2:
            flags += cv2.CALIB_FIX_K2
        if n_dist_coeff == 0:
            flags += cv2.CALIB_FIX_K1
    _, A, k, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=src2_m,
        imagePoints=dest2_m,
        imageSize=(image_width, image_height),
        distCoeffs=np.zeros(6),
        cameraMatrix=np.zeros((3, 3)),
        flags=flags,
    )

    Rt = []
    if mask is not None:
        for s, d in zip(src2, dest2):
            retval, rvec, tvec = cv2.solvePnP(s, d, A, k)
            R = util.rodrigues2matrix(rvec)
            Rt.append(np.concatenate([R, tvec.reshape(3, 1)], axis=1))
    else:
        for rvec, tvec in zip(rvecs, tvecs):
            R = util.rodrigues2matrix(rvec)
            Rt.append(np.concatenate([R, tvec.reshape(3, 1)], axis=1))

    if n_dist_coeff > 0:
        return A, Rt, k.flatten()[:n_dist_coeff].tolist()
    else:
        return A, Rt


def find_full_model(src, dest, n_dist_coeff, image_width, image_height, ransac=False):
    N_images = len(src)

    # initial homographies
    H = [find_homography(s, d) for s, d in zip(src, dest)]
    errs = [
        np.mean(
            np.linalg.norm(
                src_i - apply_homography(dest_i, np.linalg.pinv(H_i)), axis=1
            )
        )
        for H_i, dest_i, src_i in zip(H, dest, src)
    ]

    # init intrinsics and extrinsics
    if ransac:
        thresh = 1.1 * np.array(errs)  # reprojection error must not become worse..

        def model_f(mask):
            return _calibrate_camera_wrap(src, dest, image_width, image_height, 0, mask)

        def err_f(model):
            A, Rt = model[0], model[1]
            dest_t = [
                apply_pinhole(dest_i, Rt_i, A, [0], inverse=True)
                for dest_i, Rt_i in zip(dest, Rt)
            ]
            err = [
                np.mean(np.linalg.norm(src_i - dest_t_i, axis=1))
                for src_i, dest_t_i in zip(src, dest_t)
            ]
            return np.array(err) < thresh

        model, inlier, _ = util.ransac(N_images, 3, model_f, err_f)
        A, Rt = model[0], model[1]

        Rt, src, dest = util.filter_lists([Rt, src, dest], inlier)

    # estimate using all inliers / all views
    A, Rt, k = _calibrate_camera_wrap(
        src, dest, image_width, image_height, n_dist_coeff
    )

    if ransac:
        return Rt, A, k, inlier
    else:
        return Rt, A, k


def warp_image(img, transform, x0, y0, xstep, ystep, w, h, method="linear"):
    xi, yi = np.meshgrid(np.arange(x0, x0 + w, xstep), np.arange(y0, y0 + h, ystep))
    samples = np.stack((xi.flatten(), yi.flatten()), axis=1)
    samples = transform(samples)
    v = scipy.interpolate.interpn(
        (
            np.arange(img.shape[0], dtype=np.float32),
            np.arange(img.shape[1], dtype=np.float32),
        ),
        img,
        np.fliplr(samples),
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    return v.reshape((xi.shape[0], xi.shape[1]))


def warp_image_patches(img, transform, x0s, y0s, xstep, ystep, ws, hs, method="linear"):
    result = []
    for x0, y0, w, h in zip(x0s, y0s, ws, hs):
        result.append(warp_image(img, transform, x0, y0, xstep, ystep, w, h, method))
    return result


class Transform(ABC):
    """Base transform object"""

    @abstractmethod
    def __init__(self, src, dest, image_width=0, image_height=0, ransac=False):
        pass

    @abstractmethod
    def __call__(self, coords):
        pass

    @abstractmethod
    def inv(self):
        pass

    @abstractproperty
    def mask(self):
        pass

    @abstractproperty
    def reprojection_error(self):
        pass

    @abstractmethod
    def get_parameter_handles(self):
        pass

    @abstractmethod
    def update_parameters_by_handles(self):
        pass

    @abstractproperty
    def valid(self):
        pass

    def mean_scale(self):
        a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        at = self(a)
        s1 = (
            1 / 2 * (np.linalg.norm(at[1] - at[0]) + np.linalg.norm(at[3] - at[2])) / 10
        )
        s2 = (
            1 / 2 * (np.linalg.norm(at[2] - at[1]) + np.linalg.norm(at[0] - at[3])) / 10
        )
        s = 1 / 2 * (s1 + s2)
        d = np.round(s / 8)  # ensure that it is divisible by 8
        return float(d * 8)


class IdentityTransform(Transform):
    def __init__(self):
        pass

    def __call__(self, coords):
        return coords

    def inv(self):
        return self

    @property
    def mask(self):
        return None

    @property
    def reprojection_error(self):
        return 0.0

    def get_parameter_handles(self):
        return None

    def update_parameters_by_handles(self):
        return None

    def valid(self):
        return True


class MultiTransform(Transform):
    @abstractmethod
    def __getitem__(self, i):
        pass

    @abstractmethod
    def __len__(self):
        pass


class HomographyTransform(Transform):
    def __init__(
        self, src, dest, image_width=0, image_height=0, ransac=False, ransac_thres=0.1
    ):
        self._image_width = image_width
        self._image_height = image_height
        self._handles = None

        if ransac:
            # initial estimate to determine scale
            H = find_homography(src, dest)
            r = apply_homography(np.array([[0, 0], [1, 1]]), H)
            scale = np.linalg.norm(r[1] - r[0])

            H, self._mask, _ = find_homography_ransac(src, dest, ransac_thres * scale)
            src = src[self._mask]
            dest = dest[self._mask]
        else:
            self._mask = None

        self._H = find_homography(src, dest)
        self._err = np.mean(
            np.linalg.norm(apply_homography(src, self._H) - dest, axis=1)
        )

    def __call__(self, coords):
        return apply_homography(coords, self._H)

    def _get_sensitivity(self, axis):
        src = np.array([[0.0, 0.0]])
        t1 = self(src)
        src[0, axis] = 1.0
        t2 = self(src)
        return np.linalg.norm([t1 - t2])

    def inv(self):
        c = copy(self)
        c._H = np.linalg.pinv(self._H)
        c._H /= c._H[2, 2]
        return c

    @property
    def mask(self):
        return self._mask

    @property
    def reprojection_error(self):
        return self._err

    def get_parameter_handles(self):
        x_sens = self._get_sensitivity(0)
        y_sens = self._get_sensitivity(1)
        dx = 1 / x_sens * self._image_width
        dy = 1 / y_sens * self._image_width
        self._handles = np.array([[-dx, dy], [dx, dy], [dx, -dy], [-dx, -dy]])
        return self(self._handles).flatten()

    def update_parameters_by_handles(self, handles):
        handles = handles.reshape(self._handles.shape)
        self._H = find_homography(self._handles, handles)

    @property
    def valid(self):
        return self._H is not None


class FullTransform(Transform):
    def __init__(
        self, src, dest, image_width=0, image_height=0, ransac=False, n_dist_coeff=1
    ):
        Rt, self._A, self._dist = find_full_model(
            [src], [dest], n_dist_coeff, image_width, image_height, ransac=ransac
        )
        self._Rt = Rt[0]
        self._is_inverse = False
        self._success = True

    def __call__(self, coords):
        if self._success:
            return apply_pinhole(
                coords, self._Rt, self._A, self._dist, self._is_inverse
            )
        else:
            return None

    def inv(self):
        if self._success:
            c = copy(self)
            c._A = self._A
            c._Rt = self._Rt
            c._is_inverse = True
            return c
        else:
            return None

    @property
    def mask(self):
        return self._mask

    @property
    def reprojection_error(self):
        return None

    def get_parameter_handles(self, extrinsic=True, intrinsic=False, distortion=False):
        params = []

        if extrinsic:
            params.append(util.matrix2rodrigues(self._Rt[:, :3]))
            params.append(self._Rt[:, 3])
        if intrinsic:
            params.append(self._A[[0, 0, 1, 1], [0, 2, 1, 2]])
        if distortion:
            params.append(self._dist)

        return np.concatenate(params, axis=0)

    def update_parameters_by_handles(
        self, handles, extrinsic=True, intrinsic=False, distortion=False
    ):
        if extrinsic:
            self._Rt[:, :3] = util.rodrigues2matrix(handles[:3])
            self._Rt[:, 3] = handles[3:6]
            handles = handles[6:] if handles.shape[0] > 6 else None
        if intrinsic:
            self._A[[0, 0, 1, 1], [0, 2, 1, 2]] = handles[:4]
            handles = handles[4:] if handles.shape[0] > 4 else None
        if distortion:
            self._dist = handles

    @property
    def valid(self):
        return self._success


class FullMultiTransform(MultiTransform):
    def __init__(
        self,
        src,
        dest,
        image_width=0,
        image_height=0,
        ransac=False,
        n_dist_coeff=1,
        fixed_intrinsics=None,
    ):
        if ransac:
            assert fixed_intrinsics is None

        if ransac:
            self._Rt, self._A, self._dist, self._mask = find_full_model(
                src, dest, n_dist_coeff, image_width, image_height, ransac=ransac
            )
        else:
            if fixed_intrinsics is None:
                self._Rt, self._A, self._dist = find_full_model(
                    src, dest, n_dist_coeff, image_width, image_height, ransac=ransac
                )
            else:
                self._A = fixed_intrinsics[0]
                self._dist = fixed_intrinsics[1]
                self._Rt = _solve_PnP_wrap(src, dest, self._A, self._dist, n_dist_coeff)
        self._is_inverse = False
        self._success = True

    def __call__(self, coords):
        if self._success:
            return [
                apply_pinhole(x, rt, self._A, self._dist, self._is_inverse)
                for x, rt in zip(coords, self._Rt)
            ]
        else:
            return None

    def inv(self):
        if self._success:
            c = copy(self)
            c._A = self._A
            c._Rt = self._Rt
            c._is_inverse = True
            return c
        else:
            return None

    def mask(self):
        return self._mask

    def reprojection_error(self):
        return None

    def __getitem__(self, i):
        res = FullTransform.__new__(FullTransform)
        res._success = self._success
        res._A = self._A
        res._dist = self._dist
        res._Rt = self._Rt[i]
        res._err = None
        res._mask = None
        res._is_inverse = self._is_inverse
        return res

    def __len__(self):
        return len(self._Rt)

    def get_subset(self, indicator):
        res = FullMultiTransform.__new__(FullMultiTransform)
        res._success = self._success
        res._A = self._A
        res._dist = self._dist
        res._Rt = [x for i, x in enumerate(self._Rt) if indicator[i]]
        res._mask = None
        res._is_inverse = self._is_inverse
        return res

    def get_parameter_handles(
        self, items, extrinsic=True, intrinsic=False, distortion=False
    ):
        res = []
        res.append(
            self[items[0]].get_parameter_handles(
                extrinsic=extrinsic, intrinsic=intrinsic, distortion=distortion
            )
        )

        if extrinsic:
            for i in items[1:]:
                res.append(
                    self[i].get_parameter_handles(
                        extrinsic=extrinsic, intrinsic=False, distortion=False
                    )
                )

        return np.concatenate(res, axis=0)

    def update_parameters_by_handles(
        self, handles, items, extrinsic=True, intrinsic=False, distortion=False
    ):
        t = [self[i] for i in items]
        l = (
            t[0]
            .get_parameter_handles(
                extrinsic=extrinsic, intrinsic=intrinsic, distortion=distortion
            )
            .shape[0]
        )
        t[0].update_parameters_by_handles(
            handles[:l], extrinsic=extrinsic, intrinsic=intrinsic, distortion=distortion
        )
        handles = handles[l:] if handles.shape[0] > l else None
        self._A = t[0]._A
        self._dist = t[0]._dist
        self._Rt[items[0]] = t[0]._Rt

        if extrinsic and len(items) > 1:
            l = handles.shape[0] // (len(items) - 1)
            for ti, i in zip(t[1:], items[1:]):
                ti.update_parameters_by_handles(handles[:l])
                self._Rt[i] = ti._Rt
                handles = handles[l:] if handles.shape[0] > l else None

    @property
    def valid(self):
        return self._success
