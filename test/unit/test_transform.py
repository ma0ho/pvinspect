import numpy as np
from pvinspect.common import transform as t
from pvinspect.common import util
from test.utilities import assert_equal


def test_distort_undistort():
    x = np.random.rand(1000, 2) - 0.5
    k = [0.1]

    xt = t.distort(x, k)
    xtt = t.distort_inverse(xt, k)
    err = np.linalg.norm(x - xtt)
    assert err <= 1e-8

    k[-1] *= -1
    xt = t.distort(x, k)
    xtt = t.distort_inverse(xt, k)
    err = np.linalg.norm(x - xtt)
    assert err <= 1e-8


def test_find_homography_exact():
    H = np.array([[0.8, 0, 1.2], [0.2, 2.0, 10], [0, 0, 1]])

    src = np.random.rand(4, 3)
    src[:, -1] = 1
    dest = H.dot(src.T).T
    dest[:, :2] /= dest[:, 2].reshape(-1, 1)

    H2, residual = t.find_homography(src[:, :2], dest[:, :2], True)
    dest2 = H2.dot(src.T).T
    dest2 /= dest2[:, 2].reshape(-1, 1)
    err = np.linalg.norm((dest - dest2).flatten())

    assert residual < 0.1
    assert err < 1e-8
    assert np.linalg.norm(H.flatten() - H2.flatten()) < 1e-8


def test_find_homography_l2():
    H = np.array([[0.8, 0, 1.2], [0.2, 2.0, 10], [0, 0, 1]])

    src = np.random.rand(100, 3)
    src[:, -1] = 1
    dest = H.dot(src.T).T
    dest[:, :2] /= dest[:, 2].reshape(-1, 1)
    err_term = np.random.uniform(-0.2, 0.2, dest[:, :2].shape)
    dest[:, :2] += err_term

    H2, residual = t.find_homography(src[:, :2], dest[:, :2], True)
    dest2 = H2.dot(src.T).T
    dest2 /= dest2[:, 2].reshape(-1, 1)
    err = np.mean(np.linalg.norm(dest - dest2, ord=2, axis=1))
    err_term_norm = np.mean(np.linalg.norm(err_term, ord=2, axis=1))

    assert np.abs(err - err_term_norm) < 0.05
    assert np.abs(err - residual) < 0.5


def test_apply_homography():
    H = np.array([[0.8, 0, 1.2], [0.2, 2.0, 10], [0, 0, 1]])
    src = np.random.rand(1000, 3)
    src[:, -1] = 1
    dest = H.dot(src.T).T
    dest[:, :2] /= dest[:, 2].reshape(-1, 1)

    res = t.apply_homography(src[:, :2].reshape(10, 100, 2), H)
    res = res.reshape(1000, 2)
    err = np.mean(np.linalg.norm(dest[:, :2] - res[:, :2], axis=1))

    assert err < 1e-8


def test_find_homography_ransac():
    H = np.array([[0.8, 0, 1.2], [0.2, 2.0, 10], [0, 0, 1]])

    src = np.random.rand(10, 3)
    src[:, -1] = 1
    dest = H.dot(src.T).T
    dest[:, :2] /= dest[:, 2].reshape(-1, 1)
    dest[:2] += 1  # 2 outlier

    H2, is_inlier, N = t.find_homography_ransac(src[:, :2], dest[:, :2], 0.01)
    dest2 = H2.dot(src.T).T
    dest2 /= dest2[:, 2].reshape(-1, 1)
    err = np.linalg.norm((dest[3:] - dest2[3:]).flatten())

    assert err < 1e-8
    assert np.sum(is_inlier) == 8
    assert not is_inlier[0]
    assert not is_inlier[1]


# def test_full_model():
#    N_imgs = 5
#
#    Rt = [util.random_rotation_translation_matrix_3d() for i in range(N_imgs)]
#    dist = [0.05]
#    A = np.array([[900, 0.00, 255], [0, 1200, 255], [0, 0, 1]])
#    src = [np.random.rand(40, 2) * 10 for i in range(N_imgs)]
#    dest = [
#        t.apply_homography(t.distort(t.apply_homography(x, y[:, [0, 1, 3]]), dist), A)
#        for x, y in zip(src, Rt)
#    ]
#
#    Rt2, A2, dist2 = t.find_full_model(src, dest, 1, image_width=500, image_height=500)
#
#    assert np.mean(np.abs(A - A2)) < 2
#    assert_equal(dist[0], dist2[0], 1e-1)
#    for i in range(N_imgs):
#        assert np.mean(np.abs(Rt[i] - Rt2[i])) < 2


def test_homography_transform():
    H = np.array([[20, 0, 10], [0, 30, 15], [0, 0, 1]]).astype(np.float64)
    src = np.random.rand(20, 2)
    dest = t.apply_homography(src, H)

    transform = t.HomographyTransform(src, dest)
    src_t = transform(src)
    src_res = transform.inv()(src_t)

    err = np.mean(np.linalg.norm(src - src_res, axis=1))
    assert err < 1e-8


def test_full_transform():
    A = np.array([[1200, 0, 500], [0, 900, 450], [0, 0, 1]])
    Rt = util.random_rotation_translation_matrix_3d()
    dist = [0.5]

    src = np.random.rand(20, 2)
    dest = t.apply_pinhole(src, Rt, A, dist)

    transform = t.FullTransform(src, dest, 1000, 900, n_dist_coeff=1)
    src_t = transform(src)
    src_res = transform.inv()(src_t)

    err = np.mean(np.linalg.norm(src - src_res, axis=1))
    assert err < 1e-2

    err = np.mean(np.linalg.norm(src_t - dest, axis=1))
    assert err < 1e-2


def test_full_multi_transform():
    A = np.array([[1200, 0, 500], [0, 900, 450], [0, 0, 1]])
    Rt = [util.random_rotation_translation_matrix_3d() for i in range(5)]
    dist = [0.15]
    src = [np.random.rand(20, 2) for i in range(5)]
    dest = [t.apply_pinhole(s, r, A, dist) for s, r in zip(src, Rt)]

    transform = t.FullMultiTransform(src, dest, 1000, 900, ransac=False, n_dist_coeff=1)
    src_t = [t(x) for t, x in zip(transform, src)]
    src_res = [t.inv()(x) for t, x in zip(transform, src_t)]

    errs = [np.mean(np.linalg.norm(x - y, axis=1)) for x, y in zip(src_res, src)]
    assert np.mean(errs) < 1e-4

    transform = transform.inv()
    src_res = [t(x) for t, x in zip(transform, src_t)]

    errs = [np.mean(np.linalg.norm(x - y, axis=1)) for x, y in zip(src_res, src)]
    assert np.mean(errs) < 1e-4


def test_distortion_model_is_consistent():
    k1 = 0.08
    k2 = -0.01
    src = np.random.rand(20, 2)
    r = np.linalg.norm(src, axis=1).reshape(-1, 1)
    src_dist = src + src * k1 * r ** 2 + src * k2 * r ** 4
    src_dist2 = t.distort(src, [k1, k2])

    err = np.mean(np.abs(src_dist - src_dist2))
    assert err < 1e-8
