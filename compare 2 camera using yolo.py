import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # or use a different model as needed

# Initialize cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(1)  # Second camera

while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Failed to grab frame from one or both cameras")
        break
    
    # Run YOLO detection on both frames
    results1 = model(frame1)
    results2 = model(frame2)
    
    # Draw bounding boxes and labels on frames
    frame1 = results1[0].plot()
    frame2 = results2[0].plot()
    
    # Display the frames side by side
    combined_frame = np.hstack((frame1, frame2))
    cv2.imshow('Camera Comparison', combined_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
