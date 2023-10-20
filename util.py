import numpy as np
import cv2
import time

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value
    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit
    
    # if (165 <= hue <= 180) or (0 <= hue <= 15):  # Handle red and purple hues
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    # else:
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)


    # return lowerLimit, upperLimit


start_time = time.time()
def read_fps(frame_count):
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    fps = frame_count / elapsed_time
    fps = frame_count / elapsed_time
    return fps
