from ultralytics import YOLO
import cv2

model = YOLO("y8best.pt")

# webcam
results = model.predict(source=0, show=True)
