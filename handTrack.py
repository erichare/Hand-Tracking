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

def area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def sameRect(rect1, rect2):
    test1 = (((abs(rect1[0] - rect2[0]) < 25) and (abs(rect1[1] - rect2[1]) < 25)))
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
		
    return (X_[0], X_[1], X[2])

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
        cv2.imshow('hist', img)
    
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
                        print "OOOPS. too far apart"

                except: print "FAILED: Non-positive sizes"; self.selection = None; self.tracking = False

if __name__ == "__main__":

    folder = sys.argv[1]
    fnames = os.listdir(folder)
    fnames.sort()
    fnames = [f for f in fnames if f.find('image_') >= 0]
    n = len(fnames)/2

    sameCounts = []
    camShifter = []
    plotPoints = []
    colors = []
    circles = []
    for i in range(NUM_HANDS):
        camShifter.append(App())
        plotPoints.append([[], [], []])
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        circles.append([])
        sameCounts.append(0)

    image = None

    
    startTime = time()
    FPS = 0
    lastI = 0
    valCal = 40
    hueCal = 50
    for i in range(n):
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

        originalImage = image.copy()
        original = image.copy()

        shp = (image.shape[0], image.shape[1])

        red = np.zeros(shp, dtype='uint8')
        green = np.zeros(shp, dtype='uint8')
        blue = np.zeros(shp, dtype='uint8')

        hue = np.zeros(shp, dtype='uint8')
        sat = np.zeros(shp, dtype='uint8')
        val = np.zeros(shp, dtype='uint8')

        hands = np.zeros(shp, dtype='uint8')

        cv2.split(image, (blue, green, red))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.split(hsv, (hue, sat, val))

        #cv2.imshow('Live', image)
        #cv2.waitKey()
        #cv2.imshow('Hue', hue)
        #cv2.waitKey()
        #cv2.imshow('Saturation', sat)
        #cv2.imshow('Value', val)
        #cv2.waitKey()

        ret, hue = cv2.threshold(hue, hueCal, 255, cv2.THRESH_TOZERO)
        ret, hue = cv2.threshold(hue, 255 - hueCal, 255, cv2.THRESH_TOZERO_INV)
        ret, hue = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY_INV)

        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_TOZERO) #set to 0 if <= 64, otherwise leave as is
        sat = cv2.equalizeHist(sat)
        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_BINARY) #set to 0 if <= 64, otherwise 255

        ret, val = cv2.threshold(val, valCal, 255, cv2.THRESH_TOZERO) #set to 0 if <= 50, otherwise leave as is
        val = cv2.equalizeHist(val)
        ret, val = cv2.threshold(val, valCal, 255, cv2.THRESH_BINARY) #set to 0 if > 204, otherwise leave as is


        #cv2.imshow('Saturation threshold', sat)
        #cv2.waitKey()
        #cv2.imshow('Hue threshold', hue)
        #cv2.waitKey()
        #cv2.imshow('Val threshold', val)
        #cv2.waitKey()

        one = cv2.multiply(hue, sat)
        hands = cv2.multiply(one, val)

        #smooth + threshold to filter noise
        hands = cv2.blur(hands, (13, 13))
        ret, hands = cv2.threshold(hands, 200, 255, cv2.THRESH_BINARY)

        #filteredImages[0][hands == 0] = 0

        #cv2.imshow('Original', image)
        #cv2.waitKey()
        cv2.imshow('Hands', hands)
        #cv2.waitKey()
        #cv2.imshow('Filtered', filteredImage)
        #cv2.waitKey()
        #cv2.imwrite('out.jpg', hands)
        #cv2.waitKey()

        handContours = findHands(hands)
        rects = []
        factor = 15
        for j in range(len(handContours)):
            handCnt = handContours[j]
            rect = cv2.boundingRect(handCnt)
            newRect = (max(rect[0] - factor, 0), max(rect[1] - factor, 0), min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0]))
            if (area(newRect) > 2000 and (len(rects) == 0 or not sameRect(newRect, rects[0]))):
                rects.append(newRect)
                cv2.rectangle(originalImage, (max(rect[0] - factor, 0), max(rect[1] - factor, 0)), (min(rect[0] + rect[2] + factor, shp[1]), min(rect[1] + rect[3] + factor, shp[0])), (255, 255, 0))
        rects.sort()
        #rects.reverse()
            
        for j in range(len(rects)):
            im = np.zeros(shp + (3,), dtype = "uint8")
            im[rects[j][1] : rects[j][3], rects[j][0] : rects[j][2], :] = 1
            out = cv2.multiply(image, im)
            camShifter[j].initialize(original, out, rects[j], colors[j])
 
        print "#### Rects ####"
        print rects
        print "#### ####"

        for j in range(len(camShifter)):
            camShift = camShifter[j]
            camShift.run()
            if (camShift.selection == camShift.prevSelection):
                sameCounts[j] = sameCounts[j] + 1
            else:
                sameCounts[j] = 0
            if (sameCounts[j] == 4):
                camShift.selection = None; camShift.tracking = False
                print "OOOOPS. Same one!"
                sameCounts[j] = 0
            elif (camShift.tracking):
                if (valCal < 70 and (len(circles[j]) % 20 == 19)):
                    valCal = valCal + 5
                if (hueCal > 30 and (len(circles[j]) % 20 == 19)):
                    hueCal = hueCal - 5
                try: cv2.ellipse(originalImage, camShift.track_box, camShift.color, 2)
                except: 
                    print "Could not draw ellipse"

                print "#### RotatedRect TrackBox 1 ####"
                try: print camShifter[j].track_box
                except: print "Could not print trackbox"
                print "#### ####"

                print "#### PLOT ####"
                try: 
                    point = worldPoint(camShift.track_box[0][0], camShift.track_box[0][1], depth[camShift.track_box[0][0], camShift.track_box[0][1]])
                    print point
                    
                    if (point[0] > 0.5):
                        for k in range(3):
                            plotPoints[j][k].append(point[k])
                except: print ""
                print "#### ####"

            try: circles[j].append((int(camShift.track_box[0][0]), int(camShift.track_box[0][1])))
            except: print ""
            drawCircles(image, circles[j], color = colors[j])

            print "VALCAL IS", valCal
            print "HUECAL IS", hueCal
            print "I is", len(circles[j])

        cv2.imshow('Live', image)
        cv2.imshow('camshift', originalImage)
        cv2.waitKey()

    #fig = plt.figure()

    # this connects each of the points with lines
    #ax = p3.Axes3D(fig)
    # plot3D requires a 1D array for x, y, and z
    # ravel() converts the 100x100 array into a 1x10000 array
    #for i in range(len(camShifter)):
    #    mplColor = (float(camShifter[i].color[2]) / 255, float(camShifter[i].color[1]) / 255, float(camShifter[i].color[0]) / 255)
    #    ax.scatter3D(plotPoints[i][0], plotPoints[i][1], plotPoints[i][2], color = mplColor)
    #plt.show()

    for j in range(3):
        for i in range(len(camShifter)):
            mplColor = (float(camShifter[i].color[2]) / 255, float(camShifter[i].color[1]) / 255, float(camShifter[i].color[0]) / 255)
            time = range(len(plotPoints[i][j]))
            plt.plot(time, plotPoints[i][j], color = mplColor)

        plt.show()

