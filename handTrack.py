import cv2
import numpy as np
import sys

if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    
    shp = (image.shape[0], image.shape[1])
    hue = np.zeros(shp, dtype='uint8')
    sat = np.zeros(shp, dtype='uint8')
    val = np.zeros(shp, dtype='uint8')
    hands = np.zeros(shp, dtype='uint8')

    hsv = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)
    cv2.split(hsv, (hue, sat, val))

    #cv2.imshow('Live', image)
    #cv2.imshow('Hue', hue)
    #cv2.imshow('Saturation', sat)

    ret, hue = cv2.threshold(hue, 50, 255, cv2.THRESH_TOZERO) #set to 0 if <= 50, otherwise leave as is
    ret, hue = cv2.threshold(hue, 204, 255, cv2.THRESH_TOZERO_INV) #set to 0 if > 204, otherwise leave as is
    ret, hue = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY_INV) #set to 255 if = 0, otherwise 0
    
    ret, sat = cv2.threshold(sat, 34, 255, cv2.THRESH_TOZERO) #set to 0 if <= 34, otherwise leave as is
    sat = cv2.equalizeHist(sat)
    ret, sat = cv2.threshold(sat, 34, 255, cv2.THRESH_BINARY) #set to 0 if <= 34, otherwise 255

    #cv2.imshow('Saturation threshold', sat)
    #cv2.imshow('Hue threshold', hue)

    hands = cv2.multiply(hue, sat)

    #smooth + threshold to filter noise
    hands = cv2.blur(hands, (13, 13))
    ret, hands = cv2.threshold(hands, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow('Original', image)
    cv2.waitKey()
    cv2.imshow('Hands', hands)
    cv2.imwrite('out.jpg', hands)
    cv2.waitKey()

