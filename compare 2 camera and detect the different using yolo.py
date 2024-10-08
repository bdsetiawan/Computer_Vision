import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

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
    
    # Calculate the difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Create a combined frame: original frames side by side on top, difference at the bottom
    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((diff, np.zeros_like(diff)))  # Add blank space to match width
    combined_frame = np.vstack((top_row, bottom_row))
    
    # Add labels
    cv2.putText(combined_frame, "Camera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_frame, "Camera 2", (frame1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined_frame, "Difference", (10, frame1.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the combined frame
    cv2.imshow('Camera Comparison and Difference', combined_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
