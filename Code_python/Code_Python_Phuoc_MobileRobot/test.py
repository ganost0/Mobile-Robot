import time
import cv2
import numpy as np
import os
import serial
# hardware work
from imutils.video import FPS
from imutils.video import WebcamVideoStream
ArduinoSerial = serial.Serial('Com7', 9600,timeout=0.2)
# time.sleep(1)

# camera = cv2.VideoCapture(0).start()
camera = WebcamVideoStream(0).start()
fps = FPS().start()

i = 0
quay = 0
def Forward():
    ArduinoSerial.write(('6').encode())

 # Robot Reverse
def Reverse():
    ArduinoSerial.write(('7').encode())


# Robot Turn Right\
def Turnright():
    ArduinoSerial.write(('5').encode())

# Robot Turn Left
def Turnleft():
    ArduinoSerial.write(('8').encode())

 # Robot Stop
def Stop():
    ArduinoSerial.write(('0').encode())

# Image analysis work
def segment_colour(frame):  # returns only the red colors in the frame
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv_roi, np.array([160, 160, 10]), np.array([190, 255, 255]))
    ycr_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask_2 = cv2.inRange(ycr_roi, np.array((0., 165., 0.)), np.array((255., 255., 255.)))

    mask = mask_1 | mask_2
    kern_dilate = np.ones((8, 8), np.uint8)
    kern_erode = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kern_erode)  # Eroding
    mask = cv2.dilate(mask, kern_dilate)  # Dilating
    # cv2.imshow('mask',mask)
    return mask


def find_blob(blob):  # returns the red colored circle
    largest_contour = 0
    cont_index = 0
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > largest_contour):
            largest_contour = area

            cont_index = idx
            # if res>15 and res<18:
            #    cont_index=idx

    r = (0, 0, 2, 2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])

    return r, largest_contour


def target_hist(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv_img], [0], None, [50], [0, 255])
    return hist

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
    frame = camera.read()
    # // thời gian chò tin hiẹu gui xuong
    # time.sleep(0.1)
    # // lật ảnh
    frame = cv2.flip(frame,(1))

    # if not ret:
        # break

    global centre_x
    global centre_y
    centre_x = 0.
    centre_y = 0.
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = segment_colour(frame)  # masking red the frame
    loct, area = find_blob(mask_red)
    x, y, w, h = loct
    if (w * h) < 100:
        found = 0
    else:
        found = 1
        simg2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        centre_x = x + ((w) / 2)
        centre_y = y + ((h) / 2)
        cv2.circle(frame, (int(centre_x), int(centre_y)), 3, (0, 110, 255), -1)
        # centre_x -= 80

        # cv2.rectangle(frame,(640//2-50,480//2-50),
        #              (640//2+50,480//2+50),
        #               (255,255,255),3)

        # print(centre_x)
        # centre_y = 6 - -centre_y
        # print(centre_x, centre_y)
    initial = 10
    S = w*h
    # print('s = ',S)
    flag = 0
    # print(area)
    if (found == 0):
        # found Red
        i = i+1
        # #  quay  một lúc nêú không thấy, nó sẽ dừng 50s rồi tiếp tục quét
        if (i >= 20):
            print('Stop')
            # Stop()
            if i >= 40:
                print('turn right')
                if i >= 60:
                    print( 'ádasd')
                    # Turnleft()
                    i = 0
        # elif flag == 0:
        #     print('Turnright')
        #     # Turnleft()

        print('i =',i)
        # Detect Red
    elif (found == 1):
        if (centre_x <= (0.3*640) ):
            # print('turn right')
            Turnright()
        elif(centre_x >= (640-(640*0.3))):
            # print('Turn left')
            Turnleft()
        elif (area > initial ):
            initial2 = 58000
            if (area < initial2):
                # print('Forward')
                Forward()
            if (area >= initial2):
                quay = quay + 1
                print(quay)
                # print(' STOP 1 ')
                # if quay == 10 :
                #     print(' Turn Right')
                # elif quay >= 20:
                #     fps.stop()
                    # print(' Stop quay ra ')
                    # quay = 0
                    # if quay >= 30:
                        # quay = 0
                    # break
                # Stop()

    # cv2.rectangle(frame,(640),(480),(255,255,255),3)
    cv2.imshow("Original", frame)
    fps.update()

    #      thoi gian cho lay anh
    1000/41
    1000/1

    if (cv2.waitKey(55) & 0xff == ord('q')):
        fps.stop()
        break
print("Fps: {:.2f}".format(fps.fps()))
fps.update()
camera.stop()
cv2.destroyAllWindows()