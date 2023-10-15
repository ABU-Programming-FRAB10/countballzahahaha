import cv2
import numpy as np
from util import get_limits, read_fps
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('WIN_20231010_17_47_03_Pro.mp4')
# fps=cap.get(cv2.CAP_PROP_FPS)
#lower_range=np.array([])
#upper_range=np.array([])
red = [0,0,255]
lower_purple = np.array([100,0,100])
upper_purple = np.array([255,255,218])
frame_count = 0
grahp = []
def purple(img):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 0
    fps = read_fps(frame_count)
    grahp.append(fps)
    for c in cnts: 
        x=600
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            count += 1
                
            cv2.putText(frame,("DETECT-purpleball"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"total : {count}",(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"fps : {fps:.2f}",(300,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

def Red(img):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=[0,0,255])
    print(lowerLimit, upperLimit)
    mask=cv2.inRange(hsv,lowerLimit,upperLimit)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 0
    fps = read_fps(frame_count)
    grahp.append(fps)
    for c in cnts:
            
        x=600
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            count += 1
                
            cv2.putText(frame,("DETECT-redball"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"total : {count}",(180,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"fps : {fps:.2f}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

try:
    while True:
        ret,frame=cap.read()
        frame=cv2.resize(frame,(640,480))
        count = 0
        Red(frame)
        purple(frame)
        frame_count += 1
        fps = read_fps(frame_count)
        grahp.append(fps)
        cv2.imshow("FRAME",frame)
            
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
    plt.plot(grahp)
    plt.xlabel("frame")
    plt.ylabel("fps")
    plt.title("fps")
    plt.savefig('line_plot.png')
    plt.show()
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    plt.plot(grahp)
    plt.xlabel("frame")
    plt.ylabel("fps")
    plt.title("fps")
    plt.savefig('line_plot.png')
    plt.show()
