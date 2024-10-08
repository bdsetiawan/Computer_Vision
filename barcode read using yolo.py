import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model for barcode detection
model = YOLO('yolov8n.pt')  # You may need to train or find a specific model for barcode detection

# Initialize camera
cap = cv2.VideoCapture(1)  # Use the appropriate camera index

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run YOLO detection on the frame
    results = model(frame)
    
    # Process the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            
            # Extract the barcode region
            barcode_region = frame[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(barcode_region, cv2.COLOR_BGR2GRAY)
            
            # Use a barcode reader library (e.g., pyzbar) to decode the barcode
            # For this example, we'll just display the extracted region
            cv2.imshow('Barcode', gray)
            
            # Draw bounding box on the original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the frame with detections
    cv2.imshow('Barcode Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
