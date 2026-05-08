from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weight/yolov8n.pt') # wait of the img

results = model("images/velo.png", show=True)
cv2.waitKey(0) # unless user input dont do anything