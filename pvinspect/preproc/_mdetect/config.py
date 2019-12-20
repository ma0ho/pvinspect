# module/cell detection
GAUSSIAN_RELATIVE_SIGMA = 0.01
RANSAC_THRES = 0.1                  # 10% of cell size
OUTER_CORNER_THRESH_FACTOR = 2      # gradient must be this times the std dev of gradient to be accepted as outer corner
MEDIAN_FILTER = 3                   # size of median filter
CORNER_DETECTION_PATCH_SIZE = 1     # size of warped fragment
CROP_MARGIN = 0.5                   # margin around cropped modules (relative to cell size)

# outlier rejection
REPROJ_THRES = 0.1                  # threshold on reprojection error to reject images; 5% of cell size
USE_RANSAC = False                  # apply ransac during joint parameter estimation
