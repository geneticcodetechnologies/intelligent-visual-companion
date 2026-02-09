# where the face recognizer will be used
import face_recognition
import numpy as np

def get_face_encoding(frame):
    rgb = frame[:, :, ::-1]  # BGR → RGB
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        return None

    return encodings[0]
