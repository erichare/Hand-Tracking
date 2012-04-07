import numpy as np
import cv2
import sys
import os
import video
import string

help_message = '''USAGE: camshift.py [<video source>]

Select a bright colored object to track.

Keys:
  ESC   - exit
  b     - toggle back-projected probability visualization
'''


class App(object):
    def __init__(self, video_src):
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.video_src = video_src
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        elif self.drag_start: 
            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

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
        while True:
            lst = os.listdir(self.video_src)
            for f in lst:
                if len(f) > 7 and string.find(f, 'rgb.png') != -1:
                    path = self.video_src + '/' + f
                    self.frame = cv2.imread(path)
                    vis = self.frame.copy()
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array((0, 60, 32), dtype="uint8"), np.array((180, 255, 255), dtype="uint8"))

                    if self.selection:
                        x0, y0, x1, y1 = self.selection
                        self.track_window = (x0, y0, x1-x0, y1-y0)
                        hsv_roi = hsv[y0:y1, x0:x1]
                        mask_roi = mask[y0:y1, x0:x1]
                        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                        self.hist = hist.reshape(-1)
                        self.show_hist()
                        
                        vis_roi = vis[y0:y1, x0:x1]
                        cv2.bitwise_not(vis_roi, vis_roi)
                        vis[mask == 0] = 0

                    if self.tracking_state == 1:
                        print "Self.tracking_state on"
                        self.selection = None
                        prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                        prob &= mask
                        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                        track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                        
                        if self.show_backproj:
                            vis[:] = prob[...,np.newaxis]
                        try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                        except: print track_box
                        
                    cv2.imshow('camshift', vis)

                    ch = cv2.waitKey(50)
                    if ch == 27:
                        break
                    if ch == ord('b'):
                        print "Toggled"
                        self.show_backproj = not self.show_backproj


if __name__ == '__main__':
    video_src = sys.argv[1]
    App(video_src).run()

