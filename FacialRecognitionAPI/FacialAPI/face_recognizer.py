import face_recognition
import cv2
import numpy as np


def get_face_encoding(frame):
    # Convert BGR (OpenCV) → RGB (face_recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb)

    if len(face_locations) == 0:
        return None

    # Get encodings
    encodings = face_recognition.face_encodings(rgb, face_locations)

    if len(encodings) == 0:
        return None

    return encodings[0]
