#===============================================================================
# util.py
# 
# Utility functions.
#===============================================================================

import numpy as np
import numpy
import os
from numpy import array, dot
import cv2

# ============= Transform-specific ================= #
	
camAlpha = 570.3
camCenter = array([ [320,], [240,]])

def invertTransform(arr):
	R = arr[0:3, 0:3];
	t = arr[0:3, 3];
	
	Rprime = R.transpose();
	tprime = -numpy.dot(Rprime,t)
	
	Tprime = numpy.eye(4)
	Tprime[0:3, 0:3] = Rprime
	Tprime[0:3, 3] = tprime
	return Tprime


def projectPoints(X_wor, Twc):
	if X_wor.ndim > 1:
		m = X_wor.shape[1]
	else:
		m = 1

	#X_wor = .3*array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]);
	X_hom = numpy.ones((4,m));
	if m > 1:
		X_wor = X_wor.transpose();
		X_hom[0:3, 0:m] = X_wor;
	else:
		X_hom[0] = X_wor[0]
		X_hom[1] = X_wor[1]
		X_hom[2] = X_wor[2]
		
	X_cam = dot(Twc, X_hom);
	X_im = array([ X_cam[0,:]/X_cam[2,:], X_cam[1,:]/X_cam[2,:] ]);
	X_draw = X_im*camAlpha + numpy.tile(camCenter, (1, X_im.shape[1]));

	return X_draw

def depthBackgroundSubtract(depth, staticMap=None):
	if not staticMap:
		staticMap = cv2.imread("staticmapCV2.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
	nodepthmask = cv2.compare(depth, np.array(10), cv2.CMP_LE)
	depth[nodepthmask != 0] = 10000

	return cv2.compare(depth, staticMap, cv2.CMP_LT)

def segmentAndMask(depth):

    depth_final = depth.copy()

    # only keep foreground
    foregroundMask = depthBackgroundSubtract(depth)

    backgroundMask = np.zeros(depth.shape, dtype="uint8")
    backgroundMask = cv2.bitwise_not(foregroundMask)

    depth_final[backgroundMask != 0] = 0

    # mask out the pixels with no depth
    noDepthMask = cv2.compare(depth, np.array(0), cv2.CMP_LE)
    depth_final[noDepthMask != 0] = 0

    return depth_final

def buildMinMap(folder, needFlip=False):
	debug = False

	fnames = os.listdir(folder)
	fnames = [fn for fn in fnames if fn.find('dep.png') > 0]

	shp = None
	if fnames:
		frame = cv2.imread(os.path.join(folder, fnames[0]), -1)
		shp = frame.shape

	big = 65535
	minMap = np.ones(shp, 'uint16')*big
		
	for fn in fnames:
		if debug:
			print minMap.max(), minMap.min()
	
		path = os.path.join(folder, fn)
		frame = cv2.imread(path, -1)
				
		frame[frame <= 0] = big
		minMap = np.minimum(minMap, frame)
		
	minMap[minMap == big] = 0
	return minMap

