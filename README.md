# Car-Detection-OpenCV

**Vehicle detection is one of the widely used features by companies and organizations these days. This technology uses computer vision to detect different types of vehicles in a video or real-time via a camera. It finds its applications in traffic control, car tracking, creating parking sensors and many more.**

**In this repository, we will learn how to build a car detecting system in python for both recorded and live cam streamed videos.**

# Installation

First of all make sure you have Python installed in your system.

Secondly you have to install pip. It can be installed easily using command prompt.

    python -m pip install

The last requirement is the OpenCV module. It can be installed using pip.

    pip install opencv-python

# Detecting cars in Video/Image

Importing necessary Python & OpenCV libraries.

    import cv2
    
### Capturing Video

**videoCapture** is used to capture videos.

For capturing video in the real time using external camera-    

      cap = cv2.VideoCapture(1)
      
For capturing or importing video from the saved files-

      cap = cv2.VideoCapture('video path')
 
### ***INPUT VIDEO-***

https://user-images.githubusercontent.com/86667690/127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4


### Using prebuilt XML classifiers

These are the pre-trained Classifiers that can be directly used.

There are various Cascade classifiers available that can be used according to the requirement.

    car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')
      
### Loop Condition

Applying Loop Condition because of the continuity of the video.

    while True:

Capturing frame by frame.

    ret, frame = cap.read()

### Converting image to gray scale

Converting video into gray scale of each frames. 

The detection effectively works only on grayscale images/frames. So it is important to convert the colour frames to grayscale.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

### Detecting cars

**detectMultiScale** function is used to detect the cars. It takes 3 arguments — the input image/frame, scaleFactor and minNeighbours. **scaleFactor** specifies how much the image size is reduced with each scale. **minNeighbours** specifies how many neighbors each candidate rectangle should have to retain it.

    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
   
**cars** contains a list of coordinates for the rectangular regions where cars were found. We use these coordinates to draw the rectangles in our image/video.

    for (x,y,w,h) in cars:
    
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

Displaying the video in real time scenario.

        cv2.imshow('video', frame)
        
        crop_img = frame[y:y+h,x:x+w]
        
Stopping the program if the Q key is pressed.

    if cv2.waitKey(25) & 0xFF == ord('q'):
        
        break
    
Release the VideoCapture object.

    cap.release()

Closing all the frames.
       
    cv2.destroyAllWindows()
    
### ***OUTPUT RESULT-***

![Output_image (3)](https://user-images.githubusercontent.com/86667690/127769547-9aff5cea-4778-423e-b410-b4ebc18f0011.png)

# Counting cars in Video/Real Time

Importing necessary Python & OpenCV libraries.

    import cv2

    import numpy as np
    
### Capturing Video

**videoCapture** is used to capture videos.

For capturing video in the real time using external camera-    

      cap = cv2.VideoCapture(1)
      
For capturing or importing video from the saved files-

      cap = cv2.VideoCapture('video path')    
    
https://user-images.githubusercontent.com/86667690/127770208-c6794c92-a82c-4871-a6d3-0d8490f02e18.mp4

### Video frame rate

Adjusting Frame rate of the Video
    
    fps = cap.set(cv2.CAP_PROP_FPS,1)
           
### Setting some prior values
Setting minimum contour width.
   
    min_contour_width=40  #40

Setting minimum contour height.
        
    min_contour_height=40 
    
    offset=10   
    
    line_height=550  
    
    matches =[]
    
    cars=0 
    
### Defining Function

    def get_centroid(x, y, w, h):

      x1 = int(w / 2)
      
      y1 = int(h / 2)

      cx = x + x1
      
      cy = y + y1
      
      return cx,cy
    


    cap.set(3,1920)
    
    cap.set(4,1080)
     
 ..........currently under work sorry for the inconvience
    
    
