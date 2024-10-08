import cv2
import numpy as np
from pyzbar.pyzbar import decode

def process_frame(frame):
    # Convert to grayscale for better barcode detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect barcodes in the image using pyzbar
    barcodes = decode(gray)

    for barcode in barcodes:
        # Get the bounding box for the barcode
        (x, y, w, h) = barcode.rect
        # Draw a rectangle around the barcode
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract barcode data and type
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # Display the barcode data and type on the frame
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected {barcode_type}: {barcode_data}")

    return frame

def main():
    # Initialize the video capture object
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame to detect and decode barcodes
        frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Barcode/QR code Scanner', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
