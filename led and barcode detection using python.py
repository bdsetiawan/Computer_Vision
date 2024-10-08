import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Initialize the camera
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for LED detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # LED detection (simple threshold-based approach)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Adjust this value based on LED size
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "LED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Barcode detection
    barcodes = decode(frame)
    for barcode in barcodes:
        # Extract barcode information
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # Draw rectangle around the barcode
        points = barcode.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            cv2.polylines(frame, [hull], True, (255, 0, 0), 2)
        else:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (255, 0, 0), 2)

        # Put barcode information on the image
        x, y, _, _ = barcode.rect
        cv2.putText(frame, f"{barcode_data} ({barcode_type})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('LED and Barcode Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
