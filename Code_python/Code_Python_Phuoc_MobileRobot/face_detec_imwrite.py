
import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import serial
import  os

from utils import label_map_util
from utils import visualization_utils_color as vis_util

from imutils.video import FPS
from imutils.video import WebcamVideoStream
#  Tran
# ArduinoSerial = serial.Serial('Com7', 9600,timeout=0.2)
# time.sleep(1)
path = 'F:\pythonE\Face-detection\curent\img_face_detect'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './proto/label_map.pbtxt'

NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Send to Arduino
#  Robot Forward
# def Forward():
#     ArduinoSerial.write(('6').encode())
#
#  # Robot Reverse
# def Reverse():
#     ArduinoSerial.write(('7').encode())
#
#
# # Robot Turn Right\
# def Turnright():
#     ArduinoSerial.write(('8').encode())
#
# # Robot Turn Left
# def Turnleft():
#     ArduinoSerial.write(('5').encode())
#
#  # Robot Stop
# def Stop():
#     ArduinoSerial.write(('0').encode())

def face_detection():
    q = 0
    # Load Tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # Actual detection.
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Start video stream
    cap = WebcamVideoStream(0).start()
    fps = FPS().start()
    d = 0
    while True:

        frame = cap.read()
        filename = "./Face_%d.jpg"%d
        # khung hinh 480x640
        # (h,w) = frame.shape[:2]
        # print(h,w)
        frame = cv2.flip(frame,1)
        #  dầu vào mô hình thường 300x300 theo thứ tự GRB
        # frame = cv2.resize(frame,(160,140))

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        expanded_frame = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num_c) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: expanded_frame})
        # Visualization of the detection
        coords = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.8)
        # print(coords)
        try:
            if (coords != None):
                for coord in coords  :
                    # q = q+1
                # print('q:', q)
                # if q == 5 :
                    (y1, y2, x1, x2, accuracy, classificaion) = coord
                    w = x2-x1
                    h = y2-y1
                    Cir =x1+ ((w)/2)
                    # print(Cir)
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 1)
                    if not coord:
                        # print(coord)
                        # print("abc")
                        print("looking for questure")
                    else:
                        # print(coord)
                        S = w*h
                    # print(S)
            else:
                S = 1
            print(S)
        except:
            pass
        if (S < 10):
            found =0

        else:
            found =1
        flag =0
        if (found==0):
            print('Stop')
            q=0
            # Stop()

        # elif (found==1):
            # q = q + 1
            if (S > 40000):
                print('Reverse')
                q=0
                # Reverse()

            elif (S < 40000):
                d = d + 1
                cv2.imwrite(os.path.join(path, filename), frame)
                print('OK OK OK')
                # Stop()
            # if min_score_thresh >= 90:
            print(d)
            if (S < 10000):
                q=q+1
                #  quét khuono mặt trong 20s thì mới bám
                print("Q",q)
                if q >= 30:
                    print('Forward')
                    print('q:', q )
                # Forward()

            if (Cir < (0.3*640)):
                print('Turn Left')
                q=0
                # Turnleft()

            elif (Cir > (640-(0.3*640))):
                print('Turn Right')
                q=0
                # Turnright()
        #
        # # cv2.rectangle(frame,(640),(480),(255,255,255),3)
        cv2.imshow('Detection', frame)
        fps.update()

        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            break

    print("Fps: {:.2f}".format(fps.fps()))
    fps.update()
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_detection()
