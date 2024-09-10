import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model for face detection
#model.download()
# Load the YOLOv8 model for object detection
model = YOLO('yolov8n.pt')  # This model can detect various objects, including people

#model = YOLO('yolov8n-face.pt')

# Open a connection to the camera
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('Face Detection with YOLOv8', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
