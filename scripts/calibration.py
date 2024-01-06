import numpy as np
import cv2
import copy
import glob

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


# Load calibration images
left = glob.glob('data/Stereo_calibration_images/left-*.png')
right = glob.glob('data/Stereo_calibration_images/right-*.png')
left = sorted(left)
right = sorted(right)

# Set parameters of objects matrix
row, col = 6, 9
objp = np.zeros((row*col, 3), np.float32)
objp[:,:2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)

# Store objects coordinate and 2D points coordinate
obj=[]
impl= []
impr = []

for imgl, imgr in zip(left, right):
    # Left load each image and find corners
    imgL = cv2.imread(imgl)
    retl, corners_l = cv2.findChessboardCorners(imgL, (col, row))
    im_l = copy.deepcopy(imgL)
    cv2.drawChessboardCorners(im_l, (col,row), corners_l, retl)
    # Right load each image and find corners
    imgR = cv2.imread(imgr)
    retr, corners_r = cv2.findChessboardCorners(imgR, (col,row))
    im_r = copy.deepcopy(imgR)
    cv2.drawChessboardCorners(im_r, (col,row), corners_r, retr)

    # Store size of pictures
    h, w = imgR.shape[:2]
    # Store corresponding points coordinats in world frame and image frame
    if retl == True & retr == True:
        obj.append(objp)
        impl.append(corners_l)
        impr.append(corners_r)

retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(obj, impl, (w,h), None, None)
retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(obj, impr, (w,h), None, None)

# Set flags
flags =0
flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
flags |= cv2.CALIB_SAME_FOCAL_LENGTH
flags |= cv2.CALIB_RATIONAL_MODEL
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5
flags |= cv2.CALIB_FIX_ASPECT_RATIO

# Set criteria
cri = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Calibration
ret, mtxl, distl, mtxr, distr, R, T, E, F = cv2.stereoCalibrate(obj, impl, impr, mtxl, distl, mtxr, distr, (w,h), criteria=cri, flags=flags)

# Recitification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxl, distl, mtxr, distr, (w,h), R, T)

# Save parameters
np.save('parameters/mtxl.npy', mtxl)
np.save('parameters/mtxr.npy', mtxr)
np.save('parameters/distl.npy', distl)
np.save('parameters/distr.npy', distr)
np.save('parameters/R1.npy', R1)
np.save('parameters/R2.npy', R2)
np.save('parameters/P1.npy', P1)
np.save('parameters/P2.npy', P2)

# Remapping test
mapl1, mapl2 = cv2.initUndistortRectifyMap(mtxl, distl, R1, P1, (w,h),cv2.CV_32FC1)
mapr1, mapr2 = cv2.initUndistortRectifyMap(mtxr, distr, R2, P2, (w,h), cv2.CV_32FC1)

im_l = cv2.imread(left[0])
im_r = cv2.imread(right[0])

r_im_l = cv2.remap(im_l, mapl1, mapl2, cv2.INTER_LINEAR)
r_im_r = cv2.remap(im_r, mapr1, mapr2, cv2.INTER_LINEAR)

comb_im_l = cv2.hconcat([im_l, r_im_l])
comb_im_r = cv2.hconcat([im_r, r_im_r])

cv2.imshow("Left image", comb_im_l)
cv2.imshow("Right image", comb_im_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
