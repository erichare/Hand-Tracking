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

NUM_HANDS = 2
DEBUG = False

def area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def sameRect(rect1, rect2):
    test1 = (((abs(rect1[0] - rect2[0]) < 35) and (abs(rect1[1] - rect2[1]) < 35)))
    test2 = ((rect1[0] > rect2[0]) and (rect1[2] < rect2[2]) and (rect1[1] > rect2[1]) and (rect1[3] < rect2[3]))
    test3 = ((rect1[0] < rect2[0]) and (rect1[2] > rect2[2]) and (rect1[1] < rect2[1]) and (rect1[3] > rect2[3]))
    return (test1 or test2 or test3)

def worldPoint(j, i, d):
    px = camCenter[0][0]
    py = camCenter[1][0]
    x = (j - px)/camAlpha
    y = (i - py)/camAlpha
    z = d
	
    X = np.array([x, y, d, 1])
    X_ = np.dot(Tcw, X.transpose())
		
    return (X_[0], X_[1], X_[2])

def findHands(hands, num = NUM_HANDS):
    contours, hierarchy = cv2.findContours(hands, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return heapq.nlargest(num, contours, key = len)

def drawCircles(image, circles, color = (0, 255, 0)):
    for circ in circles:
        cv2.circle(image, circ, 2, color, -1)
    

class App(object):
    def __init__(self):
        cv2.namedWindow('camshift')
        self.tracking = False
        self.selection = None
        self.prevSelection = None
        self.reDraw = False
        self.track_box = None

    def initialize(self, original, image, selection, color):
        self.image = image
        self.color = color
        if not self.tracking:
            self.selection = selection

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        #cv2.imshow('hist', img)
    
    def run(self):
        if self.selection:
            vis = self.image.copy()
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0, 60, 32), dtype="uint8"), np.array((180, 255, 255), dtype="uint8"))

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

                    if self.selection[0] > 640 or self.selection[0] < 0 or self.selection[1] > 480 or self.selection[1] < 0:
                        self.selection = None; self.tracking = False

                    if ((abs(selection[0] - self.selection[0]) < 100) and (abs(selection[1] - self.selection[1]) < 100)):
                        self.prevSelection = self.selection
                        self.selection = selection
                        self.track_box = track_box
                    else:
                        self.selection = None; self.tracking = False
                        self.reDraw = True

                except:     
                    if DEBUG:
                        print "FAILED: Non-positive sizes"; self.selection = None; self.tracking = False

def getHands(folder, image, depth, showImages = False):

    camShifter = []
    colors = []
    circles = []
    for i in range(NUM_HANDS):
        camShifter.append(App())
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        circles.append([])

    valCal = 60

    mask = ~np.bool8(cv2.imread(os.path.join(folder, 'mask.png'), -1))
    tablemodel = util.buildMinMap(os.path.join(folder, 'table'))

    originalImage = image.copy()
    original = image.copy()

    shp = (image.shape[0], image.shape[1])

    hue = np.zeros(shp, dtype='uint8')
    sat = np.zeros(shp, dtype='uint8')
    val = np.zeros(shp, dtype='uint8')

    hands = np.zeros(shp, dtype='uint8')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.split(hsv, (hue, sat, val))

    if DEBUG:
        cv2.imshow('Hue', hue)
        cv2.imshow('Saturation', sat)
        cv2.imshow('Value', val)

    newDep = cv2.convertScaleAbs(cv2.absdiff(depth, tablemodel))
    ret, depThresh = cv2.threshold(newDep, 15, 255, cv2.THRESH_BINARY)
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

    outputOld = cv2.multiply(image, im)
    ret, output = cv2.threshold(outputOld, 0, 255, cv2.THRESH_BINARY)

    output1 = np.zeros(shp, dtype='uint8')
    output2 = np.zeros(shp, dtype='uint8')
    output3 = np.zeros(shp, dtype='uint8')
    cv2.split(output, (output1, output2, output3))

    sat = cv2.inRange(sat, np.array((68)), np.array((200)))
    hue = cv2.inRange(hue, np.array((0)), np.array((22)))
    ret, val = cv2.threshold(val, valCal, 255, cv2.THRESH_TOZERO)
    val = cv2.equalizeHist(val)
    ret, val = cv2.threshold(val, valCal, 255, cv2.THRESH_BINARY)


    if DEBUG:
        cv2.imshow('Saturation threshold', sat)
        cv2.imshow('Hue threshold', hue)
        cv2.imshow('Val threshold', val)

    one = cv2.multiply(hue, sat)
    two = cv2.multiply(one, val)
    hands = cv2.multiply(two, output1)

    #smooth + threshold to filter noise
    hands = cv2.blur(hands, (13, 13))
    ret, hands = cv2.threshold(hands, 200, 255, cv2.THRESH_BINARY)
    
    handContours = findHands(hands)
    rects = []
    factor = 15
    for j in range(len(handContours)):
        handCnt = handContours[j]
        rect = cv2.boundingRect(handCnt)
        newRect = (max(rect[0] - factor, 0), max(rect[1] - factor, 0), min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0]))
        if (area(newRect) > 2000 and (len(rects) == 0 or not sameRect(newRect, rects[0]))):
            rects.append(newRect)
            if DEBUG:
                cv2.rectangle(originalImage, (max(rect[0] - factor, 0), max(rect[1] - factor, 0)), (min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0])), (255, 255, 0))
    rects.sort()
        
    for j in range(len(rects)):
        im = np.zeros(shp + (3,), dtype = "uint8")
        im[rects[j][1] : rects[j][3], rects[j][0] : rects[j][2], :] = 1
        out = cv2.multiply(image, im)
        val = j
        if (len(rects) == 1 and not camShifter[val].tracking):
            val = (0 if rects[0][0] < 270 else 1)
            
        camShifter[val].initialize(original, out, rects[j], colors[val])

    if DEBUG:
        print "#### Rects ####"
        print rects
        print "#### ####"

    for j in range(len(camShifter)):
        camShift = camShifter[j]
        camShift.run()
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
            point = worldPoint(camShift.track_box[0][0], camShift.track_box[0][1], depth[camShift.track_box[0][0], camShift.track_box[0][1]])
            print "Position: ", camShift.track_box[0]
            print "Size: ", camShift.track_box[1]
            print "Orientation: ", camShift.track_box[2]
            print "World Point: ", point

        except: print ""
        print "#### ####"

    if (showImages):
        cv2.imshow('Hands', hands)
        cv2.imshow('camshift', originalImage)
        cv2.waitKey()

    handList = [None, None]
    for j in range(NUM_HANDS):
        if (camShifter[j].track_box is not None):
            handList[(j + 1) % 2] = camShifter[j].track_box[0]

    return handList
