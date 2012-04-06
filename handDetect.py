import numpy as np
import cv2

if __name__ == '__main__':
    import sys
    import getopt

    args, img_args = getopt.getopt(sys.argv[1], '', ['cascade='])
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_hand.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)

    img = cv2.imread(sys.argv[1])
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(100, 100))
    print rects
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0))
        cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)
    cv2.imshow("Output", img)
    cv2.waitKey()
