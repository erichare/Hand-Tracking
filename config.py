#Input configuration info

screenWidth = 1280
screenHeight = 800
videoWidth = 640
videoHeight = 480

cameraConfigFile = "./SamplesConfig.xml"

showFPS = True # print FPS to the console
DRAGGINGTHRESHOLD = 30 #distance you  need to drag before it will be considered not to be noise

#ryder
#touch event parameters
FINGER_SIZE = 50
NOISE_SIZE = 4
SMOOTHING_FACTOR = 15
THRESHOLD_FACTOR = 60
#/ryder

HAND_THRESHOLD_HIGH = 50     # in mm
HAND_THRESHOLD_LOW = 5     # in mm

# shared parameters
saveWorldsConcurrently = True
BACKTRACK_LIM = 1
DISPLAY_FLAG = False
saveWorldsAfter = False

WORLD_SNAPSHOT_LIM = 1
NOMINAL_MIN_DEPTH = 500
NOMINAL_MAX_DEPTH = 2000
VNI_GRASPED_SIM_THRESH = .83
PROBATION_LENGTH = 6
MIN_BLOBSIZE = 1000
VNI_DIFF_THRESH = .88
OCC_DIFF_THRESH = .8
DEPTH_NOISE_MARGIN = 10
OBJ_DEPTH_NOISE_MARGIN = 10
HAND_SIZE = 80
HAND_SMOOTHING_PARAM = 2
SLEEVES = True

