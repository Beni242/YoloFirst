from ultralytics import YOLO
import cv2
import cvzone # display all detection
import math


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

while True:
    sucess, img = cap.read()
    img = cv2.resize(img, (940, 700)) # for videos redimention
    results = model(img, stream=True)

    # design the boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box

             x1, y1, x2, y2 = box.xyxy[0]

             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

             #print(x1, y1, x2, y2)

                # with open cv

             #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # with cvzone
             w, h = x2-x1, y2-y1
             cvzone.cornerRect(img, (x1, y1, w,h))

            # confidance

             conf = math.ceil((box.conf[0]*100))/100

            # Class Name

             cls = int (box.cls[0])

             cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)  # display a box text up the zone

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break