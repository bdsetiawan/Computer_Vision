
import cv2
from pyzbar.pyzbar import decode

# Open a connection to the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Detect barcodes in the frame
    barcodes = decode(frame)
    
    # Process detected barcodes
    for barcode in barcodes:
        # Extract barcode information
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        
        # Ignore if the barcode is tilted to the left or right
        if barcode.polygon[0][1] != barcode.polygon[1][1]:
            continue
        
        # Draw rectangle around the barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put barcode information on the image
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Detected {barcode_type}: {barcode_data}")
    
    # Display the resulting frame
    cv2.imshow('Barcode Reader', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
