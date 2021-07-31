# importing libraries
import cv2
import numpy as np

# capturing or reading video
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('cars.mp4')

# adjusting frame rate
fps = cap.set(cv2.CAP_PROP_FPS,1)


# minimum contour width
min_contour_width=40  #40

# minimum contour height
min_contour_height=40  #40
offset=10   #10
line_height=550  #550
matches =[]
cars=0

# defining a function
def get_centroid(x, y, w, h):

    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx,cy
    return [cx, cy]


cap.set(3,1920)
cap.set(4,1080)

if cap.isOpened():
    ret,frame1 = cap.read()
else:
    ret = False
ret,frame1 = cap.read()
ret,frame2 = cap.read()

while ret:
    d = cv2.absdiff(frame1,frame2)
    grey = cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey,(5,5),0)

    ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(th,np.ones((3,3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours,h = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue
        cv2.rectangle(frame1,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)

        cv2.line(frame1, (0, line_height), (1200, line_height), (0,255,0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1,centroid, 5, (0,255,0), -1)
        cx,cy= get_centroid(x, y, w, h)
        for (x,y) in matches:
            if (line_height + offset) > y > (line_height - offset):
                cars=cars+1
                matches.remove((x,y))
                print(cars)

    cv2.putText(frame1, "Total Vehicles Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)

    #cv2.drawContours(frame1,contours,-1,(0,0,255),2)


    cv2.imshow("OUTPUT" , frame1)
    #cv2.imshow("Difference" , th)
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret , frame2 = cap.read()

#print(matches)
cv2.destroyAllWindows()
cap.release()