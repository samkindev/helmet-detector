from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture("./test2.mp4")
# cap = cv2.VideoCapture(0)

model = YOLO('helmet-model-150.pt')

classNames = ["Helmet"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2) 

            bbox = x1, y1, x2 - x1, y2 - y1

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)

            if conf > 0.4:
                cvzone.cornerRect(img, bbox)
                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(50, y1)), scale=0.7, thickness=1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)