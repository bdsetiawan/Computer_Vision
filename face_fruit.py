import cv2
import mediapipe as mp

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize object detection for fruits
mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, model_name='Cup')
mp_drawing = mp.solutions.drawing_utils

# Open a connection to the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Convert the frame to RGB for object detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects in the frame
    result = objectron.process(rgb_frame)
    
    # Draw landmarks on the detected objects
    if result.detected_objects:
        for detected_object in result.detected_objects:
            mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)
    
    # Display the resulting frame
    cv2.imshow('Face and Fruit Detection', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
