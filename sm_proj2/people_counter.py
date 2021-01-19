import cv2
import numpy as np

capture = cv2.VideoCapture('test2.mp4')

ret, frame1 = capture.read()
ret, frame2 = capture.read()


while capture.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
    for contour in contours:
        print("Número de carros: %d"  %len(contours))
        (x,y,w,h) = cv2.boundingRect(contour) 

        if cv2.contourArea(contour) < 9000 or cv2.contourArea(contour) > 12000:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
      
    
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = capture.read()

    if cv2.waitKey(40) == 27:
        break
  
cv2.destroyAllWindows()
capture.release()
