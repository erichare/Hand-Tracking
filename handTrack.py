import cv2
import numpy as np
import handTrackModule
import sys
import os
import util
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from time import clock, time

showFPS = False

# This function attempts to calibrate the value based on the hand video
# However, it doesn't seem to work too well...
def calibrateVal(handFolder):
    print handFolder
    fnames = os.listdir(handFolder)
    fnames.sort()
    fnames = [f for f in fnames if f.find('image_') >= 0]
    n = len(fnames)/2
    i = 0
    while (i < n):
        depPath = os.path.join(folder, 'image_'+str(i)+'_dep.png')
        imgPath = os.path.join(folder, 'image_'+str(i)+'_rgb.png')

        image = cv2.imread(imgPath)
        depth = cv2.imread(depPath, -1)

        shp = (image.shape[0], image.shape[1])

        hue = np.zeros(shp, dtype='uint8')
        sat = np.zeros(shp, dtype='uint8')
        val = np.zeros(shp, dtype='uint8')

        hands = np.zeros(shp, dtype='uint8')

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.split(hsv, (hue, sat, val))

        print cv2.mean(val)[0]
        i = i + 1;

# Main routine to test the handTrackModule
if __name__ == "__main__":

    # Get the name of all the images
    folder = sys.argv[1]
    fnames = os.listdir(folder)
    fnames.sort()
    fnames = [f for f in fnames if f.find('image_') >= 0]
    n = len(fnames)/2

    # Store some values in order to keep track of FPS
    if (showFPS):
        startTime = time()
        FPS = 0
        lastI = 0

    # Get our plot points ready
    timePoints = [[], []]
    plotPoints = [[[], [], []], [[], [], []]]

    # Create the mask and table model
    mask = ~np.bool8(cv2.imread(os.path.join(folder, 'mask.png'), -1))
    tablemodel = util.buildMinMap(os.path.join(folder, 'table'))

    i = 0
    waitAmount = 5
    
    handList = None
    camShifter = None
    colors = None

    # Loop until we are out of images
    while (i < n):
        
        print "Processing Frame ", i

        # Show the FPS if desired
        if (showFPS):
            if (time() - startTime > 1):
                FPS = i - lastI
                startTime = time()
                lastI = i
            print "#### FPS ####"
            print FPS
            print "#### ####"

        # Grab the depth and the RGB images
        depPath = os.path.join(folder, 'image_'+str(i)+'_dep.png')
        imgPath = os.path.join(folder, 'image_'+str(i)+'_rgb.png')
        image = cv2.imread(imgPath)
        depth = cv2.imread(depPath, -1)

        # Call the module and store the results
        handList, camShifter, colors, originalImage = handTrackModule.getHands(image, depth, camShifter, colors, mask, tablemodel)
        
        # Plot the points if they are returned
        for j in range(len(handList)):
            hand = handList[j]
            if hand:
                point = handTrackModule.worldPoint(hand[0], hand[1], depth[hand[1], hand[0]])
                if (abs(point[1]) > 0.5):
                    for k in range(3):
                            plotPoints[(j + 1) % 2][k].append(point[k])
                    timePoints[(j + 1) % 2].append(i)


        cv2.imshow("Image", originalImage)
        key = cv2.waitKey(waitAmount)

        ## Handle keystrokes
        # 's' = stop
        # 'a' = rewind
        # 'd' = fastforward
        # 'q' = quit
        ##
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


    # Attempt to draw the graph
    fig = plt.figure()

    for j in range(3):
        for i in range(len(camShifter)):
            mplColor = (float(colors[i][2]) / 255, float(colors[i][1]) / 255, float(colors[i][0]) / 255)
            plt.plot(timePoints[i], plotPoints[i][j], color = mplColor)

        plt.show()

