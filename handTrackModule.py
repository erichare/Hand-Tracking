import cv2
import numpy as np
import sys
import os
import string
import util
import matplotlib.pyplot as plt
import heapq
import random
import mpl_toolkits.mplot3d.axes3d as p3
from time import clock, time

# Set the number of hands possible in a frame (default is 2)
# Debug mode will show cutoff values for HSV/depth, etc
NUM_HANDS = 2
DEBUG = False

# Prepare for plotting points
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

# Returns the area of a rectangle
def area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

# Returns true if the two rectangles are the 'same'... that is, they are bounding to the same blob
def sameRect(rect1, rect2):
    test1 = (((abs(rect1[0] - rect2[0]) < 40) and (abs(rect1[1] - rect2[1]) < 40)))
    test2 = ((rect1[0] > rect2[0]) and (rect1[2] < rect2[2]) and (rect1[1] > rect2[1]) and (rect1[3] < rect2[3]))
    test3 = ((rect1[0] < rect2[0]) and (rect1[2] > rect2[2]) and (rect1[1] < rect2[1]) and (rect1[3] > rect2[3]))
    test4 = (((abs(rect1[2] - rect2[2]) < 40) and (abs(rect1[3] - rect2[3]) < 40)))
    return (test1 or test2 or test3 or test4)

# Convert a point to a worldpoint
def worldPoint(j, i, d):
    px = camCenter[0][0]
    py = camCenter[1][0]
    x = (d * (j - px))/camAlpha
    y = (d * (i - py))/camAlpha
    z = d
	
    X = np.array([x, y, d, 1])
    X_ = np.dot(Tcw, X.transpose())
		
    return (X_[0], X_[1], X_[2])

# Grabs the num largest blobs
def findHands(hands, num = NUM_HANDS):
    contours, hierarchy = cv2.findContours(hands, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return heapq.nlargest(num, contours, key = len)

# Draws circles at the given point on the image
def drawCircles(image, circles, color = (0, 255, 0)):
    for circ in circles:
        cv2.circle(image, circ, 2, color, -1)

# Defines the Camshift class
class CamShiftTracker(object):

    # Initialize a new camshift tracker
    def __init__(self):
        self.tracking = False
        self.selection = None
        self.prevSelection = None
        self.reDraw = False
        self.track_box = None

    # If the object is not currently tracking, set a new selection window
    def initialize(self, original, image, selection, color, meanVal):
        self.image = image
        self.color = color
        self.meanVal = meanVal
        if not self.tracking:
            self.selection = selection

    # Show the color histogram of the tracked objects
    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        if DEBUG:        
            cv2.imshow('hist', img)
    
    # Primary camshift algorithm
    def run(self):

        # Only run the algorithm if we have made a rectangular selection
        if self.selection:

            # Mask out pixels that cannot be human skin
            vis = self.image.copy()
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0, 54, self.meanVal - 20), dtype="uint8"), np.array((22, 210, 255), dtype="uint8"))

            # If we are not currently tracking, calculate the color histogram of the selection
            if not self.tracking:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                self.hist = hist.reshape(-1)
                self.show_hist()
                
                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
                self.tracking = True

            # We have a color histogram.  Calculate the back projection, then store the new selection
            if self.tracking:
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

                if DEBUG:
                    print "#### CamShift Selection ####"
                    print self.selection
                    print "#### ####"

                try: 
                    track_box, selection = cv2.CamShift(prob, self.selection, term_crit)
                    self.reDraw = False

                    # Lots of error cases to consider.  Just make a new selection if we run into one
                    if (self.track_box == track_box and self.selection == selection):
                        self.selection = None; self.tracking = False; self.track_box = None

                    elif self.selection[0] > 640 or self.selection[0] < 0 or self.selection[1] > 480 or self.selection[1] < 0:
                        self.selection = None; self.tracking = False; self.track_box = None

                    elif track_box[0] < 0.01 or track_box[1] < 0.01:
                        self.selection = None; self.tracking = False; self.track_box = None

                    elif ((abs(selection[0] - self.selection[0]) < 100) and (abs(selection[1] - self.selection[1]) < 100)):
                        self.prevSelection = self.selection
                        self.selection = selection
                        self.track_box = track_box
                    else:
                        self.selection = None; self.tracking = False; self.track_box = None
                        self.reDraw = True

                except:     
                    if DEBUG:
                        print "FAILED: Non-positive sizes"; self.selection = None; self.tracking = False

# Primary function to return the location of two hands
def getHands(image, depth, camShifter = None, colors = None, mask = None, tablemodel = None, showImages = False):

    # Make sure the mask and table model are available.  Otherwise, try to create them
    if (mask is None or tablemodel is None):
        mask = ~np.bool8(cv2.imread(os.path.join(folder, 'mask.png'), -1))
        tablemodel = util.buildMinMap(os.path.join(folder, 'table'))

    # If there is not camShifter currently running, we create a new one and start from scratch
    if (camShifter is None):
        camShifter = []
        colors = []
        for i in range(NUM_HANDS):
            camShifter.append(CamShiftTracker())
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # Same for colors... choose random colors if there are not passed in
    if (colors is None):
        colors = []
        for i in range(NUM_HANDS):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # Make copies of the original image
    originalImage = image.copy()
    original = image.copy()
    shp = (image.shape[0], image.shape[1])

    # Split into HSV
    hue = np.zeros(shp, dtype='uint8')
    sat = np.zeros(shp, dtype='uint8')
    val = np.zeros(shp, dtype='uint8')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.split(hsv, (hue, sat, val))

    hands = np.zeros(shp, dtype='uint8')

    # If we have debug enabled, show the images
    if DEBUG:
        cv2.imshow('Hue', hue)
        cv2.imshow('Saturation', sat)
        cv2.imshow('Value', val)

    # Find the contours of the depth segmentation that intersect the top of the screen
    # This is a bit of a slow process unfortunately
    newDep = cv2.convertScaleAbs(cv2.absdiff(depth, tablemodel))
    ret, depThresh = cv2.threshold(newDep, 12, 255, cv2.THRESH_BINARY)
    depConts, depHier = cv2.findContours(depThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    depRects = []
    for cont in depConts:
        rect = cv2.boundingRect(cont)
        newRect = (rect[1], rect[0], rect[0] + rect[2], rect[1] + rect[3])
        if (area((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])) > 2000):
            depRects.append(newRect)
    depRects.sort()

    im = np.zeros(shp + (3,), dtype = "uint8")
    for j in range(NUM_HANDS):
        if (len(depRects) > j and depRects[j][0] < 20):
            depHand = (depRects[j][1], 0, depRects[j][2], depRects[j][3])
            cv2.rectangle(image, (depHand[0], depHand[1]), (depHand[2], depHand[3]), (255, 0, 0))

            im[depHand[1] : depHand[3], depHand[0] : depHand[2], :] = 1

    # Multiply the image by the binary depth segment
    outputOld = cv2.multiply(image, im)
    ret, output = cv2.threshold(outputOld, 0, 255, cv2.THRESH_BINARY)
    output1 = np.zeros(shp, dtype='uint8')
    output2 = np.zeros(shp, dtype='uint8')
    output3 = np.zeros(shp, dtype='uint8')
    cv2.split(output, (output1, output2, output3))

    meanVal = int(round(cv2.mean(val)[0])) - 30

    # Segment by HSV within acceptable ranges of human skin color
    sat = cv2.inRange(sat, np.array((68)), np.array((200)))
    hue = cv2.inRange(hue, np.array((0)), np.array((22)))
    val = cv2.inRange(val, np.array((meanVal)), np.array((255)))

    # Show the thresholds if debug is enabled
    if DEBUG:
        cv2.imshow('Saturation threshold', sat)
        cv2.imshow('Hue threshold', hue)
        cv2.imshow('Val threshold', val)

    # Multiply all the thresholds to obtain our final hand image
    one = cv2.multiply(hue, sat)
    two = cv2.multiply(one, val)
    hands = cv2.multiply(two, output1)

    # Smooth + threshold to filter noise
    hands = cv2.blur(hands, (13, 13))
    ret, hands = cv2.threshold(hands, 200, 255, cv2.THRESH_BINARY)
    
    # Find the hands by selecting the two largest blobs
    handContours = findHands(hands)
    rects = []
    factor = 20

    # Loop over each of the two blobs
    for j in range(len(handContours)):
        handCnt = handContours[j]

        # Bound a rectangle to the blob
        # Note that we make the rectangle a bit larger to make camshift work a little better
        rect = cv2.boundingRect(handCnt)
        newRect = (max(rect[0] - factor, 0), max(rect[1] - factor, 0), min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0]))

        # As long as the area of the rectangle meets a threshold, and it is not too similar to another, append it
        # if debug is enabled, we draw the rectangle
        if (area(newRect) > 2000 and (len(rects) == 0 or not sameRect(newRect, rects[0]))):
            rects.append(newRect)
            if DEBUG:
                cv2.rectangle(originalImage, (max(rect[0] - factor, 0), max(rect[1] - factor, 0)), (min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0])), (255, 255, 0))

    # Initialize a camshift tracker for each hand.
    # Note that if one is already tracking, initialize does nothing useful
    rects.sort()
    for j in range(len(rects)):

        # Filter out all pixels outside the rectangle found above
        im = np.zeros(shp + (3,), dtype = "uint8")
        im[rects[j][1] : rects[j][3], rects[j][0] : rects[j][2], :] = 1
        out = cv2.multiply(image, im)
        val = j
        if (len(rects) == 1 and not camShifter[val].tracking):
            val = (0 if rects[0][0] < 270 else 1)
            
        camShifter[val].initialize(original, out, rects[j], colors[val], meanVal)

    if DEBUG:
        print "#### Rects ####"
        print rects
        print "#### ####"

    # For each of the camshift trackers, run the algorithm (does nothing if there is no selection)
    for j in range(len(camShifter)):
        camShift = camShifter[j]
        camShift.run()

        # If we retrieved something useful, draw the ellipse and print information
        if (len(rects) > 0 and (camShift.tracking or camShift.reDraw)):
            try: cv2.ellipse(originalImage, camShift.track_box, camShift.color, 2)
            except: 
                print ""
            if DEBUG:
                print "#### RotatedRect TrackBox 1 ####"
                try: print camShifter[j].track_box
                except: print "Could not print trackbox"
                print "#### ####"

            whichHand = "Left" if j == 1 else "Right"
            print "#### {hand} Hand ####".format(hand = whichHand)
            try: 
                point = worldPoint(camShift.track_box[0][0], camShift.track_box[0][1], depth[camShift.track_box[0][1], camShift.track_box[0][0]])
                print "Position: ", camShift.track_box[0]
                print "Size: ", camShift.track_box[1]
                print "Orientation: ", camShift.track_box[2]
                print "World Point: ", point

            except: print ""
            print "#### ####"

    # Store the points to return
    handList = [None, None]
    for j in range(NUM_HANDS):
        if (camShifter[j].track_box is not None):
            handList[(j + 1) % 2] = camShifter[j].track_box[0]

    return handList, camShifter, colors, originalImage
