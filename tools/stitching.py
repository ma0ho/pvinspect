import numpy as np
import imutils
import cv2

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# imageB is up /down /left /right to imageA
def distinguish_direction(imageA, imageB, kpsA, kpsB, matches, status):

    #initialize the vote arrray whiiich would be used to store the direction
    vote = [0, 0, 0, 0]

    (height_B, width_B) = imageB.shape[:2]
    threshold_height = height_B // 10
    threshold_width = width_B // 10

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # imageA drew right and imageB left
            pixel_A = (int(kpsA[queryIdx][0] + width_B), int(kpsA[queryIdx][1]))
            pixel_B = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
            if np.abs(pixel_B[1] - pixel_A[1]) > threshold_height:
                if pixel_B[1] > pixel_A[1]:
                    vote[0] = vote[0] + 1
                else:
                    vote[1] = vote[1] + 1

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # imageA drew down and imageB drew up
            pixel_A = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1] + height_B))
            pixel_B = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
            if np.abs(pixel_B[0] - pixel_A[0]) > threshold_width:
                if pixel_B[0] > pixel_A[0]:
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

def match_keypoints(kpsA, kpsB, features_A, features_B, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual matches
    # use BruteForce and k-NearestNeighbor to match
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # set k to 2
    raw_matches = matcher.knnMatch(features_A, features_B, 2)

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
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        # with RANSAC it would find the optimum homography matrix
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # return the matches along with the homograpy matrix and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    print("no matches")
    return None


def stitch(images, ratio=0.55, reprojThresh=4.0):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
    (imageB, imageA) = images

    # To calculate the features
    (kpsA, features_A) = detect_and_describe(imageA)
    (kpsB, features_B) = detect_and_describe(imageB)

    # match features between the two images
    M = match_keypoints(kpsA, kpsB, features_A, features_B, ratio, reprojThresh)

    # if the match is None, then there aren't enough matched keypoints to create a panorama
    if M is None:
        return None

    # otherwise, apply a perspective warp to stitch the images together
    (matches, H, status) = M

    # distinguish the direction and to apply corresponding stitching
    direction = distinguish_direction(imageA, imageB, kpsA, kpsB, matches, status)
    if direction == UP:
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageB.shape[0] + imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    elif direction == DOWN:
        # For Down direction it is required to transform it into Up direction
        M = match_keypoints(kpsB, kpsA, features_B, features_A, ratio, reprojThresh)
        H = M[1]
        result = cv2.warpPerspective(imageB, H, (imageB.shape[1], imageB.shape[0] + imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageA.shape[1]] = imageA
    elif direction == LEFT:
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    else:
        # For Right direction it is required to transform it into Left direction
        M = match_keypoints(kpsB, kpsA, features_B, features_A, ratio, reprojThresh)
        H = M[1]
        result = cv2.warpPerspective(imageB, H,
                                     (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        result[0 : imageA.shape[0], 0:imageA.shape[1]] = imageA
    return result



