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
        self.track_box = None

    def initialize(self, image, original, selection, color):
        self.image = image.copy()
        self.original = original.copy()
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

                try: self.track_box, self.selection = cv2.CamShift(prob, self.selection, term_crit)
                except: print "FAILED: Non-positive sizes"; self.selection = False; self.tracking = False
                if (self.track_box[0][1] == 0.0):
                    self.selection = False; self.tracking = False


if __name__ == "__main__":

    folder = sys.argv[1]
    fnames = os.listdir(folder)
    fnames.sort()
    fnames = [f for f in fnames if f.find('image_') >= 0]
    n = len(fnames)/2

    camShifter = []
    plotPoints = []
    colors = []
    circles = []
    for i in range(NUM_HANDS):
        camShifter.append(App())
        plotPoints.append([[], [], []])
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        circles.append([])

    image = None

    for i in range(n):
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
        #image[:,:,:][(red > 100) & (green < 100)] = 0   

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.split(hsv, (hue, sat, val))

        #cv2.imshow('Live', image)
        #cv2.waitKey()
        #cv2.imshow('Hue', hue)
        #cv2.waitKey()
        #cv2.imshow('Saturation', sat)
        #cv2.imshow('Value', val)
        #cv2.waitKey()

        ret, hue = cv2.threshold(hue, 50, 255, cv2.THRESH_TOZERO) #set to 0 if <= 50, otherwise leave as is
        ret, hue = cv2.threshold(hue, 204, 255, cv2.THRESH_TOZERO_INV) #set to 0 if > 204, otherwise leave as is
        ret, hue = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY_INV) #set to 255 if = 0, otherwise 0

        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_TOZERO) #set to 0 if <= 64, otherwise leave as is
        sat = cv2.equalizeHist(sat)
        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_BINARY) #set to 0 if <= 64, otherwise 255

        ret, val = cv2.threshold(val, 40, 255, cv2.THRESH_TOZERO) #set to 0 if <= 50, otherwise leave as is
        val = cv2.equalizeHist(val)
        ret, val = cv2.threshold(val, 40, 255, cv2.THRESH_BINARY) #set to 0 if > 204, otherwise leave as is

        #cv2.imshow('Saturation threshold', sat)
        #cv2.waitKey()
        #cv2.imshow('Hue threshold', hue)
        #cv2.waitKey()
        #cv2.imshow('Val threshold', val)
        #cv2.waitKey()

        one = cv2.multiply(hue, sat)
        hands = cv2.multiply(one, val)

        #smooth + threshold to filter noise
        hands = cv2.blur(hands, (10, 10))
        ret, hands = cv2.threshold(hands, 170, 255, cv2.THRESH_BINARY)

        filteredImage = image.copy()
        filteredImage[hands == 0] = 0

        #cv2.imshow('Original', image)
        #cv2.waitKey()
        cv2.imshow('Hands', hands)
        #cv2.waitKey()
        cv2.imshow('Filtered', filteredImage)
        #cv2.waitKey()
        #cv2.imwrite('out.jpg', hands)
        #cv2.waitKey()

        handContours = findHands(hands)

        rects = []
        for i in range(len(handContours)):
            handCnt = handContours[i]
            rect = cv2.boundingRect(handCnt)
            newRect = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            if (area(newRect) > 100):
                rects.append(newRect)
                cv2.rectangle(originalImage, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0))
                filteredImage = filteredImage.copy()
                ogfilteredImage = filteredImage.copy()
                filteredImage[rects[i][1] : rects[i][3], rects[i][0] : rects[i][2], :] = 256
                ogfilteredImage[filteredImage != 256] = 0
                camShifter[i].initialize(ogfilteredImage, original, rects[i], colors[i])
 
        print "#### Rects ####"
        print rects
        print "#### ####"

        for i in range(len(camShifter)):
            camShift = camShifter[i]
            camShift.run()
            try: cv2.ellipse(originalImage, camShift.track_box, camShift.color, 2)
            except: print "Could not draw ellipse"

            print "#### RotatedRect TrackBox 1 ####"
            try: print camShifter1.track_box
            except: print ""
            print "#### ####"

            try: circles[i].append((int(camShift.track_box[0][0]), int(camShift.track_box[0][1])))
            except: print ""
            drawCircles(image, circles[i], color = colors[i])

            print "#### PLOT ####"
            try: 
                point = worldPoint(camShift.track_box[0][0], camShift.track_box[0][1], depth[camShift.track_box[0][0], camShift.track_box[0][1]])
                print point

                for j in range(3):
                    plotPoints[i][j].append(point[j])
            except: print ""
            print "#### ####"

        cv2.imshow('Live', image)
        cv2.imshow('camshift', originalImage)
        cv2.waitKey(10)

    for i in range(len(camShifter)):
        miniX = plotPoints[i][0].pop(plotPoints[i][0].index(min(plotPoints[i][0])))
        miniY = plotPoints[i][1].pop(plotPoints[i][1].index(max(plotPoints[i][1])))
        miniZ = plotPoints[i][2].pop(plotPoints[i][2].index(min(plotPoints[i][2])))

        while (miniX > plotPoints[i][0][plotPoints[i][0].index(min(plotPoints[i][0]))] - 1):
            miniX = plotPoints[i][0].pop(plotPoints[i][0].index(min(plotPoints[i][0])))
            miniY = plotPoints[i][1].pop(plotPoints[i][1].index(max(plotPoints[i][1])))
            miniZ = plotPoints[i][2].pop(plotPoints[i][2].index(min(plotPoints[i][2])))


        #fig = plt.figure()

        # this connects each of the points with lines
        #ax = p3.Axes3D(fig)
        # plot3D requires a 1D array for x, y, and z
        # ravel() converts the 100x100 array into a 1x10000 array
        #ax.scatter3D(plotX[i], plotY[i], plotZ[i], color = mplColor)
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #fig.add_axes(ax)

    for j in range(3):
        for i in range(len(camShifter)):
            mplColor = (float(camShifter[i].color[2]) / 255, float(camShifter[i].color[1]) / 255, float(camShifter[i].color[0]) / 255)
            time = range(len(plotPoints[i][j]))
            plt.plot(time, plotPoints[i][j], color = mplColor)

        plt.show()

