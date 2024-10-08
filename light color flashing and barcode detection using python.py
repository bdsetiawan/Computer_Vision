import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Initialize the camera
cap = cv2.VideoCapture(1)  # Use 0 for default camera, or change to the appropriate camera index

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space for LED color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for different LED colors (adjust these as needed)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine masks
    mask = mask_red | mask_green | mask_blue

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Adjust this value based on LED size
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Determine the color of the LED
            color = "Unknown"
            if cv2.countNonZero(mask_red[y:y+h, x:x+w]) > 0:
                color = "Red"
            elif cv2.countNonZero(mask_green[y:y+h, x:x+w]) > 0:
                color = "Green"
            elif cv2.countNonZero(mask_blue[y:y+h, x:x+w]) > 0:
                color = "Blue"
            
            cv2.putText(frame, f"{color} LED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    cv2.imshow('LED Color and Barcode Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
