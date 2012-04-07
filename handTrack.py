import cv2
import numpy as np
import sys
import os
import string

if __name__ == "__main__":
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
        path = sys.argv[1] + '/' + f
        if (f == ' '):
            path = path.rpartition('/')[0]
        image = cv2.imread(path)

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
        #cv2.waitKey()
        #cv2.imshow('Value', val)
        #cv2.waitKey()

        ret, hue = cv2.threshold(hue, 20, 255, cv2.THRESH_TOZERO) #set to 0 if <= 50, otherwise leave as is
        ret, hue = cv2.threshold(hue, 234, 255, cv2.THRESH_TOZERO_INV) #set to 0 if > 204, otherwise leave as is
        ret, hue = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY_INV) #set to 255 if = 0, otherwise 0

        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_TOZERO) #set to 0 if <= 34, otherwise leave as is
        sat = cv2.equalizeHist(sat)
        ret, sat = cv2.threshold(sat, 64, 255, cv2.THRESH_BINARY) #set to 0 if <= 34, otherwise 255

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
        #cv2.imshow('Hands', hands)
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

        for (val, idx) in maxVal:
            if (idx != 0):
                rect = cv2.boundingRect(contours[idx])
                cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))

        cv2.imshow('Original', image)
        cv2.waitKey(5)
        if len(sys.argv) == 1:
            lst.append(vc.read()[1])
    
    

