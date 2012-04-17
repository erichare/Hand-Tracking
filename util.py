#===============================================================================
# util.py
# 
# Utility functions.
#===============================================================================

import numpy as np
import numpy
from numpy import array, dot

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
