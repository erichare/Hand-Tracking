import cv2
import numpy as np
import handTrackModule
import sys
import os
import string
import util
import matplotlib.pyplot as plt
import heapq
import random
import mpl_toolkits.mplot3d.axes3d as p3
from time import clock, time

NUM_HANDS = 2
DEBUG = False

f = open('Tcw.txt', 'r')
aa = f.readlines();
bb = [a.strip().split() for a in aa]
arr = [ [float (b1) for b1 in b] for b in bb]
Tcw = np.array(arr)
Twc = util.invertTransform(Tcw)
camAlpha = util.camAlpha
camCenter = util.camCenter
px = camCenter[0][0]
py = camCenter[1][0]

if __name__ == "__main__":

    folder = sys.argv[1]
    fnames = os.listdir(folder)
    fnames.sort()
    fnames = [f for f in fnames if f.find('image_') >= 0]
    n = len(fnames)/2

    startTime = time()
    FPS = 0
    lastI = 0

    mask = ~np.bool8(cv2.imread(os.path.join(folder, 'mask.png'), -1))
    tablemodel = util.buildMinMap(os.path.join(folder, 'table'))

    i = 0
    waitAmount = 5
    
    handList = None
    camShifter = None
    colors = None

    while (i < n):
        
        print "Processing Frame ", i

        #if (time() - startTime > 1):
        #    FPS = i - lastI
        #    startTime = time()
        #    lastI = i
        #print "#### FPS ####"
        #print FPS
        #print "#### ####"

        depPath = os.path.join(folder, 'image_'+str(i)+'_dep.png')
        imgPath = os.path.join(folder, 'image_'+str(i)+'_rgb.png')

        image = cv2.imread(imgPath)
        depth = cv2.imread(depPath, -1)

        handList, camShifter, colors, originalImage = handTrackModule.getHands(image, depth, camShifter, colors, mask, tablemodel)

        cv2.imshow("Image", originalImage)
        key = cv2.waitKey(waitAmount)

        if (key == 115):
            key = cv2.waitKey(abs(waitAmount - 5))
            if (key == 97 or key == 100):
                waitAmount = 0
                i = i - 2
            else:
                waitAmount = 5
        elif (key == 113):
            break
        elif (key == 97):
            waitAmount = 0
            i = i - 2
        elif (key == 100):
            waitAmount = 0

        i = i + 1


    #fig = plt.figure()

    # this connects each of the points with lines
    #ax = p3.Axes3D(fig)
    # plot3D requires a 1D array for x, y, and z
    # ravel() converts the 100x100 array into a 1x10000 array
    #for i in range(len(camShifter)):
    #    mplColor = (float(camShifter[i].color[2]) / 255, float(camShifter[i].color[1]) / 255, float(camShifter[i].color[0]) / 255)
    #    ax.scatter3D(plotPoints[i][0], plotPoints[i][1], plotPoints[i][2], color = mplColor)
    #plt.show()

    #for j in range(3):
    #    for i in range(len(camShifter)):
    #        mplColor = (float(camShifter[i].color[2]) / 255, float(camShifter[i].color[1]) / 255, float(camShifter[i].color[0]) / 255)
    #        time = range(len(plotPoints[i][j]))
    #        plt.plot(time, plotPoints[i][j], color = mplColor)

    #   plt.show()

