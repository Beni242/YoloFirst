import numpy as np
from ultralytics import YOLO
import cv2
import cvzone # display all detection
import math
from sort import *


cap = cv2.VideoCapture("../video/car0fps.mp4") # for video



model = YOLO("../Yolo-Weight/yolov8n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mask = cv2.imread("mask.png")

#Tracking
tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.3)


while True:
    sucess, img = cap.read()


    img = cv2.resize(img, (940, 700)) # for videos redimention
    mask = cv2.resize(mask, (940, 700))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True) # only on the specify region it's detect it

    detections = np.empty((0,5))

    # design the boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box

             x1, y1, x2, y2 = box.xyxy[0]

             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # with cvzone
             w, h = x2-x1, y2-y1
             cvzone.cornerRect(img, (x1, y1, w,h), l=9)

            # confidance

             conf = math.ceil((box.conf[0]*100))/100

            # Class Name

             cls = int (box.cls[0])
             currurentClass = classNames[cls]

             if currurentClass == "car" or currurentClass == "truck" or currurentClass == "bus"\
                     or currurentClass == "motorbike" and conf >= 0.3:

               # cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(35, y1)),
                                   #scale=0.7, thickness=1, offset=3)  # display a box text up the zone
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 5)
                currurentArray = np.array([x1,y1,x2,y2,conf]) #list
                detections = np.vstack((detections, currurentArray))


    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w , h), l=1, rt=1, colorR=(255,0,0)) # tracker blocks
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(5, y1)),
                           scale=2, thickness=3, offset=8)

    cv2.imshow('img',img)
    cv2.imshow('imgRegion', imgRegion)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break