import cv2
import numpy as np
import sys
import os
import string

class App(object):
    def __init__(self):
        cv2.namedWindow('camshift')
        self.tracking = False

    def initialize(self, image, selection):
        self.image = image.copy()
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

            try: track_box, self.selection = cv2.CamShift(prob, self.selection, term_crit)
            except: "Non-positive sizes"
            
            try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
            except: print track_box
                
        cv2.imshow('camshift', vis)

if __name__ == "__main__":
    camShifter = App()

    image = None
    vc = None
    lst = []
    if len(sys.argv) > 1:
        try:
            lst = os.listdir(sys.argv[1])
            lst.sort()
        except:
            lst.append(' ')
    else:
        vc = cv2.VideoCapture(0)
        vc.open(0)
        vc.open(0)
        lst.append(vc.read()[1])

    for f in lst:
        image = None
        if len(sys.argv) > 1:
            path = sys.argv[1] + '/' + f
            if (f == ' '):
                path = path.rpartition('/')[0]

            image = cv2.imread(path)
        else:
            image = f

        shp = (image.shape[0], image.shape[1])
        hue = np.zeros(shp, dtype='uint8')
        sat = np.zeros(shp, dtype='uint8')
        val = np.zeros(shp, dtype='uint8')
        hands = np.zeros(shp, dtype='uint8')

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

        #cv2.imshow('Saturation threshold', sat)
        #cv2.waitKey()
        #cv2.imshow('Hue threshold', hue)
        #cv2.waitKey()

        hands = cv2.multiply(hue, sat)

        #smooth + threshold to filter noise
        hands = cv2.blur(hands, (10, 10))
        ret, hands = cv2.threshold(hands, 170, 255, cv2.THRESH_BINARY)

        #cv2.imshow('Original', image)
        #cv2.waitKey()
        cv2.imshow('Hands', hands)
        cv2.waitKey()
        #cv2.imwrite('out.jpg', hands)
        #cv2.waitKey()
        filteredImage = image.copy()
        filteredImage[hands == 0] = 0

        contours, hierarchy = cv2.findContours(hands, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        maxVal = [(0, 0), (0, 0)]
        for i in range(len(contours)):
            cnt = contours[i]
            if (maxVal[0][0] < len(cnt) or maxVal[1][0] < len(cnt)):
                maxVal = sorted(maxVal)
                maxVal.reverse()
                maxVal.pop()
                maxVal.append((len(cnt), i))

        rects = []
        for (val, idx) in maxVal:
            if (idx != 0):
                rect = cv2.boundingRect(contours[idx])
                newRect = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
                print newRect
                rects.append(newRect)
                cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))

        #cv2.imshow('Original', image)
        #cv2.waitKey()

        if len(rects) > 0 and (rects[0][2] - rects[0][0] > 40) and (rects[0][3] - rects[0][1] > 40):
            print "Running camshift"
            camShifter.initialize(image, rects[0])
            camShifter.run()

        if len(sys.argv) == 1:
            lst.append(vc.read()[1])
    
    

