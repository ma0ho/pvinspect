import numpy as np
import imutils
import cv2

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# img1 is up /down /left /right to img0
def distinguish_direction(img0, img1, kps0, kps1, matches, status):
    #initialize the vote arrray whiiich would be used to store the direction
    vote = [0, 0, 0, 0]

    (height1, width1) = img1.shape[:2]
    threshold_height = height1 // 10
    threshold_width = width1 // 10

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # img0 drew right and img1 left
            pixel0 = (int(kps0[queryIdx][0] + width1), int(kps0[queryIdx][1]))
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
            pixel0 = (int(kps0[queryIdx][0]), int(kps0[queryIdx][1] + height1))
            pixel1 = (int(kps1[trainIdx][0]), int(kps1[trainIdx][1]))
            if np.abs(pixel1[0] - pixel0[0]) > threshold_width:
                if pixel1[0] > pixel0[0]:
                    vote[2] = vote[2] + 1
                else:
                    vote[3] = vote[3] + 1
    return np.argmax(vote)


def detect_and_describe(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)


def match_keypoints(kps0, kps1, features0, features1, ratio, reproj_thresh):
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


def stitching(images, ratio=0.55, reproj_thresh=4.0):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
    (img1, img0) = images

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
    direction = distinguish_direction(img0, img1, kps0, kps1, matches, status)
    if direction == UP:
        result = cv2.warpPerspective(img0, H, (img0.shape[1], img1.shape[0] + img0.shape[0]))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
    elif direction == DOWN:
        # For Down direction it is required to transform it into Up direction
        M = match_keypoints(kps1, kps0, features1, features0, ratio, reproj_thresh)
        H = M[1]
        result = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0] + img0.shape[0]))
        result[0:img1.shape[0], 0:img0.shape[1]] = img0
    elif direction == LEFT:
        result = cv2.warpPerspective(img0, H, (img0.shape[1] + img1.shape[1], img0.shape[0]))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
    else:
        # For Right direction it is required to transform it into Left direction
        M = match_keypoints(kps1, kps0, features1, features0, ratio, reproj_thresh)
        H = M[1]
        result = cv2.warpPerspective(img1, H,
                                     (img0.shape[1] + img1.shape[1], img1.shape[0]))
        result[0 : img0.shape[0], 0:img0.shape[1]] = img0
    return result



