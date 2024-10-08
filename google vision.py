from google.cloud import vision
from google.cloud.vision_v1 import types
import io


def detect_faces(image_path):
    # Initialize the Google Vision API client
    client = vision.ImageAnnotatorClient()

    # Load the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Perform face detection on the image
    response = client.face_detection(image=image)
    faces = response.face_annotations

    print(f"Found {len(faces)} faces in the image.")

    # Process the detected faces
    for i, face in enumerate(faces):
        print(f"Face {i + 1}:")
        print(f"  Joy: {face.joy_likelihood}")
        print(f"  Sorrow: {face.sorrow_likelihood}")
        print(f"  Anger: {face.anger_likelihood}")
        print(f"  Surprise: {face.surprise_likelihood}")

        # Get face bounding box
        vertices = face.bounding_poly.vertices
        print("  Face bounds:")
        for vertex in vertices:
            print(f"    ({vertex.x},{vertex.y})")

    if response.error.message:
        raise Exception(f"{response.error.message}")


# Set your Google Application Credentials environment variable
# Replace 'your_credentials.json' with your service account key file
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "indigo-skyline-362513-3affdc5ec02d.json"

# Provide the path to the image file you want to analyze
detect_faces("C:/Users/Budi Setiawan/Downloads/face_people.jpg")
