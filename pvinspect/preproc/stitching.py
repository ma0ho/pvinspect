"""Stitching of two images of same size"""
import numpy as np
from pvinspect.data.image import *
from typing import List, Tuple
import cv2
from skimage.exposure import rescale_intensity

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def distinguish_direction(
    img: np.ndarray,
    kps0: np.ndarray,
    kps1: np.ndarray,
    matches: List[Tuple[int, int]],
    status: np.ndarray,
) -> int:
    """Find out the direction relationship between input images

        Args:
            img(np.ndarray): Input image that would be processed(considering that two image of the same size)
            kps0(np.ndarray): The obtained keypoints for the first image
            kps1(np.ndarray): The obtained keypoints for the second image
            matches(List[Tuple[int, int]]): The matches for all matched keypoints pairs
            status(np.ndarray): Showing if each keypoints would be regarded as correctly matched
        Returns:
            direction: The direction relationship between two images. From 1 - 4 means image1 (kps1) is Up/Down/Left/Right to image0 (kps0)
        """
    # initialize the vote array which would be used to store the direction
    vote = [0, 0, 0, 0]

    (height, width) = img.shape[:2]
    threshold_height = height // 10
    threshold_width = width // 10

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # img0 drew right and img1 left
            pixel0 = (int(kps0[queryIdx][0] + width), int(kps0[queryIdx][1]))
            pixel1 = (int(kps1[trainIdx][0]), int(kps1[trainIdx][1]))
            if np.abs(pixel1[1] - pixel0[1]) > threshold_height:
                if pixel1[1] > pixel0[1]:
                    vote[0] = vote[0] + 1
                else:
                    vote[1] = vote[1] + 1

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # img0 drew down and img1 drew up
            pixel0 = (int(kps0[queryIdx][0]), int(kps0[queryIdx][1] + height))
            pixel1 = (int(kps1[trainIdx][0]), int(kps1[trainIdx][1]))
            if np.abs(pixel1[0] - pixel0[0]) > threshold_width:
                if pixel1[0] > pixel0[0]:
                    vote[2] = vote[2] + 1
                else:
                    vote[3] = vote[3] + 1
    return np.argmax(vote)


def detect_and_describe(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the keypoints and correspondent features

        Args:
            image(np.ndarray): Input image that would be calculated
        Returns:
            kps: The keypoints calculated
            features: Correspondent features(1 x 128) for each keypoints
        """
    # descriptor = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.ORB_create()
    image = rescale_intensity(image, out_range=np.uint8).astype(np.uint8)
    # image = np.expand_dims(image, -1)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # plt.imshow(image)
    # plt.show()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])

    return (kps, features)


def match_keypoints(
    kps0: np.ndarray,
    kps1: np.ndarray,
    features0: np.ndarray,
    features1: np.ndarray,
    ratio: float = 0.55,
    reproj_thresh: float = 4.0,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """To match the keypoints

        Args:
            kps0(np.ndarray): The obtained keypoints for the first image
            kps1(np.ndarray): The obtained keypoints for the second image
            features0(np.ndarray): Correspondent features(1 x 128) for each keypoints in image 0
            features1(np.ndarray): Correspondent features(1 x 128) for each keypoints in image 1
            ratio(float): A parameter to control the amount and quality of the matched keypoints pairs
            reproj_thresh(float): A parameter used in calculate the homography matrix
        Returns:
            matches: The matches for all matched keypoints pairs
            H: Homography matrix
            status: Showing if each keypoints would be regarded as correctly matched
        """
    # compute the raw matches and initialize the list of actual matches
    # use BruteForce and k-NearestNeighbor to match
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # set k to 2
    raw_matches = matcher.knnMatch(features0, features1, 2)

    # only select those matches that meet the requirement
    matches = []
    # loop over the raw matches
    for m in raw_matches:
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        pts0 = np.float32([kps0[i] for (_, i) in matches])
        pts1 = np.float32([kps1[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        # with RANSAC it would find the optimum homography matrix
        (H, status) = cv2.findHomography(pts0, pts1, cv2.RANSAC, reproj_thresh)
        # return the matches along with the homograpy matrix and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    print("no matches")
    return None


def stitch_images(
    images: Tuple[Image, Image], ratio: float = 0.55, reproj_thresh: float = 4.0
) -> Image:
    """Applies stitching to a pair of partial images of a solar module.

        Args:
            images (Tuple[Image, Image]): Pair of images that should be stitched together
            ratio (float): This parameter controls the amount and quality of the matched keypoints pairs
            reproj_thresh (float): This parameter is used in calculating the homography matrix
        Returns:
            result: Stitched image
    """
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
    img1 = images[0].data
    img0 = images[1].data

    # To calculate the features
    (kps0, features0) = detect_and_describe(img0)
    (kps1, features1) = detect_and_describe(img1)

    # match features between the two images
    M = match_keypoints(kps0, kps1, features0, features1, ratio, reproj_thresh)

    # if the match is None, then there aren't enough matched keypoints to create a panorama
    if M is None:
        return None

    # otherwise, apply a perspective warp to stitch the images together
    (matches, H, status) = M

    # distinguish the direction and to apply corresponding stitching
    direction = distinguish_direction(img0, kps0, kps1, matches, status)
    if direction == UP:
        result = cv2.warpPerspective(
            img0, H, (img0.shape[1], img1.shape[0] + img0.shape[0])
        )
        result[0 : img1.shape[0], 0 : img1.shape[1]] = img1
    elif direction == DOWN:
        # For Down direction it is required to transform it into Up direction
        M = match_keypoints(kps1, kps0, features1, features0, ratio, reproj_thresh)
        H = M[1]
        result = cv2.warpPerspective(
            img1, H, (img1.shape[1], img1.shape[0] + img0.shape[0])
        )
        result[0 : img1.shape[0], 0 : img0.shape[1]] = img0
    elif direction == LEFT:
        result = cv2.warpPerspective(
            img0, H, (img0.shape[1] + img1.shape[1], img0.shape[0])
        )
        result[0 : img1.shape[0], 0 : img1.shape[1]] = img1
    else:
        # For Right direction it is required to transform it into Left direction
        M = match_keypoints(kps1, kps0, features1, features0, ratio, reproj_thresh)
        H = M[1]
        result = cv2.warpPerspective(
            img1, H, (img0.shape[1] + img1.shape[1], img1.shape[0])
        )
        result[0 : img0.shape[0], 0 : img0.shape[1]] = img0

    return Image.from_other(images[0], data=result)
