from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

results = model("img.png",show=True)
cv2.waitKey(0)