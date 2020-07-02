import numpy as np
from shapely.geometry import Polygon
from typing import List
from contextlib import contextmanager
import sys, os


def rotation_matrix_3d(rx, ry, rz):
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    return Rz.dot(Ry.dot(Rx))


def random_rotation_matrix_3d(max_rotation):
    rot = (np.random.rand(3) * 2 * max_rotation) - max_rotation
    return rotation_matrix_3d(rot[0], rot[1], rot[2])


def random_rotation_translation_matrix_3d(
    ranges=[(-20, 20), (-20, 20), (200, 400)], max_rotation=np.pi
):
    ranges = np.array(ranges)
    scales = ranges[:, 1] - ranges[:, 0]
    t = (np.random.rand(3) * scales + ranges[:, 0]).reshape(3, 1)
    r = random_rotation_matrix_3d(max_rotation)
    return np.concatenate([r, t], axis=1)


def rodrigues2matrix(r):
    angle = np.linalg.norm(r)
    r = r / angle
    W = np.array(
        [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=np.float64
    )
    r = r.reshape(-1, 1)
    rrT = r.dot(r.T)
    return (
        np.cos(angle) * np.identity(3) + (1 - np.cos(angle)) * rrT + np.sin(angle) * W
    )


def matrix2rodrigues(R):
    A = (R - R.T) / 2
    p = np.array([A[2, 1], A[0, 2], A[1, 0]])
    pn = np.linalg.norm(p)
    c = (np.trace(R) - 1) / 2

    if pn == 0.0 and c == 1.0:
        return np.zeros(3)
    elif pn == 0.0 and c == -1.0:
        Rp = R + np.identity(3)
        v = R[:, np.argmax(np.linalg.norm(R, axis=1))]
        u = v / np.linalg.norm(v)
        Su = (
            -u
            if u[0] < 0.0
            or (u[0] == 0.0 and u[1] < 0.0)
            or (u[0] == 0.0 and u[1] == 0.0 and u[2] < 0.0)
            else u
        )
        return np.pi * Su
    else:
        u = p / pn
        angle = np.arctan2(pn, c)
        return angle * u


def ransac(n_samples, n_required, model_f, err_f, p=0.99):
    is_inlier = np.zeros(n_samples, dtype=np.bool)
    num_inlier = 0
    N = np.inf
    n = 0
    best_model = None

    while n < N:
        # randomly sample n_required points
        idx = np.random.choice(n_samples, n_required, replace=False)
        mask = np.full_like(is_inlier, False)
        mask[idx] = True

        # determine model
        # returns None if failed
        tmp = model_f(mask)

        if tmp is not None:
            # compute errors
            new_inlier = err_f(tmp)
            num_new_inlier = np.sum(new_inlier)

            # accept result?
            if num_new_inlier > num_inlier:
                best_model = tmp
                is_inlier = new_inlier
                num_inlier = num_new_inlier
                w = num_inlier / n_samples
                N = np.log(1 - p) / np.log(1 - w ** n_required + 1e-8)

        n += 1

    # compute model using all inliers
    return model_f(is_inlier), is_inlier, n


def filter_lists(lists, indicator):
    res = [[v for i, v in enumerate(l) if indicator[i]] for l in lists]
    return tuple(res)


def find_file_in_list(file, files):
    idx = None
    for i, p in enumerate(files):
        if file.samefile(p):
            return i

    return None


def iou(p1, p2):
    """Calculate IOU of two shapely polygons"""
    if p1.intersects(p2):
        return p1.intersection(p2).area / p1.union(p2).area
    else:
        return 0.0


def objdetect_metrics(
    objects_label,
    objects_predicted,
    iou_thresh: List[float] = [0.6, 0.7, 0.8, 0.9, 0.99],
):
    """Calculate IOU for every pair from objects_label and objects_predicted,
    assign objects by maximum IOU and return mean IOU, precision and recall"""

    t = objects_label[0][0]
    for o in objects_label + objects_predicted:
        if o[0] != t:
            raise NotImplementedError("Method does not support different object types")

    # get rid of object type
    objects_label = [v[1] for v in objects_label]
    objects_predicted = [v[1] for v in objects_predicted]

    # set up matrix of IOUs
    max_len = max(len(objects_label), len(objects_predicted))
    ious = np.zeros((max_len, max_len), dtype=np.float64)
    for i, o1 in enumerate(objects_label):
        for j, o2 in enumerate(objects_predicted):
            ious[i, j] = iou(o1, o2)

    # for every object from objects1, find partner with maximum intersection from object2
    iou_sum = 0.0
    assignments = [dict() for _ in iou_thresh]
    for i in range(max_len):
        # find detection with maximum IOU to label i
        j = np.argmax(ious[i])

        # only assign, if j does not have a larger IOU with another label
        if np.argmax(ious[:, j]) == i:
            iou_sum += ious[i, j]

            # store assignments by iou_thresh
            for thresh_i, thresh in enumerate(iou_thresh):
                if ious[i, j] > thresh:
                    # i = label index; j = detection index
                    assignments[thresh_i][i] = j

            # "delete" partner object
            ious[:, j] = 0.0

    # calculate precision + recall
    recall = [
        len(x.keys()) / len(objects_label) if len(objects_label) > 0 else 0.0
        for x in assignments
    ]
    precision = [
        len(x.keys()) / len(objects_predicted) if len(objects_predicted) > 0 else 0.0
        for x in assignments
    ]
    return iou_sum / max_len, precision, recall


def polygon2boundingbox(polygon: Polygon):
    """Convert polygon to bounding box"""
    return Polygon.from_bounds(*polygon.bounds)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
