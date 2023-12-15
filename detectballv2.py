import cv2
import numpy as np
from util import get_limits, read_fps
import matplotlib.pyplot as plt
import math
import time
cap = cv2.VideoCapture(1)
# fps=cap.get(cv2.CAP_PROP_FPS)
#lower_range=np.array([])
#upper_range=np.array([])
red = [0,0,255]
lower_purple = np.array([92,141,0])
upper_purple = np.array([153,201,255])
l_b = np.array([24, 194, 0])
u_b = np.array([255, 255, 255])
l_r = np.array([0, 146, 0])
u_r = np.array([16, 255, 102])

frame_count = 0
grahp = []
def purple(img):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 0
    # fps = read_fps(frame_count)
    # grahp.append(fps)
    for c in cnts: 
        x=150
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            bottomLeftCornerOfText = (x, y)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
            text = "x,y" + str(bottomLeftCornerOfText)
            cv2.putText(frame,text,bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            count += 1
                
            cv2.putText(frame,("DETECT-purpleball"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"total : {count}",(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # cv2.putText(frame,f"fps : {fps:.2f}",(300,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

def blue(img):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # lowerLimit, upperLimit = get_limits(color=[0,0,255])
    # print(lowerLimit, upperLimit)
    mask=cv2.inRange(hsv,l_b,u_b)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 0
    # fps = read_fps(frame_count)
    # grahp.append(fps)
    for c in cnts:
            
        x=150
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            bottomLeftCornerOfText = (x, y)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            text = "x,y" + str(bottomLeftCornerOfText)
            count += 1
            cv2.putText(frame,text,bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
                
            cv2.putText(frame,("DETECT-blueball"),(10,140),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"total : {count}",(220,140),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

def Red(img):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # lowerLimit, upperLimit = get_limits(color=[0,0,255])
    # print(lowerLimit, upperLimit)
    mask=cv2.inRange(hsv,l_r,u_r)
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    count = 0
    # fps = read_fps(frame_count)
    # grahp.append(fps)
    for c in cnts:
            
        x=100
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            bottomLeftCornerOfText = (x, y)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            text = "x,y" + str(bottomLeftCornerOfText)
            cv2.putText(frame,text,bottomLeftCornerOfText,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            count += 1
                
            cv2.putText(frame,("DETECT-redball"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(frame,f"total : {count}",(180,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    # cv2.putText(frame,f"fps : {fps:.2f}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

end_prep = 0
fc = 0
end_prep = time.time()
fps = 0
try:
    while True:
        # start_prep = time.time()
        ret,frame=cap.read()
        # frame=cv2.resize(frame,(640,480))
        count = 0
        Red(frame)
        blue(frame)
        purple(frame)
        frame_count += 1
        # fps = math.ceil(1/(end-start))
        # fps = cvzone.FPS()
        fc += 1
        
        if time.time()-end_prep >=1:
            fps = fc/(time.time()-end_prep)
            cv2.putText(frame,f"fps : {int(fps)}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            fc = 0
            print(int(fps))
            # global grahp
            grahp.append(int(fps))
            print(grahp)
            end_prep = time.time()
        cv2.putText(frame,f"fps : {int(fps)}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        # else:
        #     cv2.putText(frame,f"fps : {int(grahp[len(grahp)])}",(270,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        
            
            
        cv2.imshow("FRAME",frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    with open(f"fps.txt", "w") as f:
        f.write(str(grahp))
        f.close
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
