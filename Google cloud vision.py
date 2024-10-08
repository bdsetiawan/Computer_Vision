import cv2
from google.cloud import vision

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Open a connection to the camera
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    content = buffer.tobytes()

    # Create an image object
    image = vision.Image(content=content)

    # Perform face detection
    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Draw rectangles around detected faces
    for face in faces:
        vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        cv2.polylines(frame, [np.array(vertices)], True, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection with Google Cloud Vision', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
