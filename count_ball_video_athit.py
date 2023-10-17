import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#Motion Tracker
# cap = cv2.VideoCapture("image/Walking.mp4")
cap = cv2.VideoCapture("ABU/WIN_20231010_17_47_03_Pro.mp4")
check , frame1 = cap.read()
check , frame2 = cap.read()
while (cap.isOpened()):
    if check == True :
        motiondiff= cv2.absdiff(frame1,frame2)
        image_rgb= cv2.cvtColor(motiondiff, cv2.COLOR_BGR2RGB)
        image_gray= cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        shifted = cv2.pyrMeanShiftFiltering(image_rgb, 0, 51)
        lab = cv2.cvtColor(shifted , cv2.COLOR_RGB2LAB)
        thresh,th1 = cv2.threshold(lab,137,255,cv2.THRESH_BINARY)
        gray_th1 = cv2.cvtColor(th1 , cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray_th1,3)
        # thresh,th3 = cv2.threshold(blur,132,255,cv2.THRESH_BINARY)

        image_new = image_rgb.copy()
        blur = cv2.medianBlur(gray_th1,3)
        ret, thresh = cv2.threshold(blur,132,255,cv2.THRESH_BINARY)

        # noise removal
        kernel = np.ones((10,10),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)

        # sure background area
        kernel = np.ones((4,4),np.uint8)
        sure_bg = cv2.dilate(opening,kernel,iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

        # Seed region
        ret,sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        num_labels, labels = cv2.connectedComponents(sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(image_rgb,markers)
        cv2.putText(frame1, " Found {} coins".format(num_labels - 1), (500, 250),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # image_rgb[markers == -1] = [0,0,255]


        #วาดสี่เหลี่ยมในวัตถุที่กำลังเคลื่อนที่
        # for contour in contours:
        #     (x,y,w,h) = cv2.boundingRect(contour)
        #     if cv2.contourArea(contour)<2500:
        #         continue
        #     cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Output",frame1)
        frame1=frame2
        print(f" Found {num_labels - 1} balls")
        check,frame2 = cap.read()
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break
    else :
        break

cap.release()
cv2.destroyAllWindows()