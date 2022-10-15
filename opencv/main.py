import cv2
import numpy as np

from urllib import request


#open video
cap = cv2.VideoCapture(0)

kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))

#Variable to count the people that croses the line
counter=0
centroid=(0,0)

#create a bground substractor
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while cap.isOpened():
    #grab the frame
    ret,frame = cap.read()
    if frame is None:
        break
    #resize the frame to display a big resolution
    frame = cv2.resize(frame, (480,480))
    #segmentate the foreground from the background
    fgmask=fgbg.apply(frame)
    #morphological operations to improve the detection of people by closing holes
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
    dilated = cv2.dilate(fgmask, None, iterations=3)

    #Bound the shape of a identified person
    #_, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours,hierachy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1000:
            continue

        #draw a rectangle that bounds a person .rectangle(image, startingPoint, endPoint, color, thickness)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #Calculate the centroid of the rectangle
        xPoint=int(x+w/2)
        yPoint=int(y+h/2)
        #coordinate of the centroid
        centroid = (xPoint, yPoint)
        #Draw the centroid of the rectangle
        cv2.circle(frame, centroid, 5, (0,0,255), 2)
        #calculate the area of the rectangle
        area = (x+w) * (y+h)
        #if the y coordinate of the centroid is inside this rank, the person has passed the line
        if 239<=xPoint<242:
            #is the area is > 156500 the rectangle is containing 2 people
            if area > 156500:
                counter = counter+2
                form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfNFtzNE0ufy8o3jK_xGDkPDalthOFu-EKPhAUdn_SCDh4epQ/formResponse?usp=pp_url&entry.1637310600={}&submit=Submit".format(counter)
                request.urlopen(form_url)
            else:
                counter = counter+1

    #draw a horizontol line that determines the represents the virtual entrance
    #If a person crosses this line following a top-down direction it will be counted
    cv2.line(frame, (0, 240), (480, 240), (0,255,255), 3)
    cv2.putText(frame, "Counted People:" + str(counter), (10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255), 2)
    #show frame
    cv2.imshow("frame", frame)
    #cv2.imshow("detected people", fgmask)
    #wait until q key is pressed down
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#relase camera and window
cap.release()
cv2.destroyAllWindows()

