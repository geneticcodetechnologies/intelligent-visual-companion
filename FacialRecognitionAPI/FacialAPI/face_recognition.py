import numpy as np
import pickle
import os

DB_PATH = "data/faces.db"

def load_faces():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def save_face(name, encoding):
    faces = load_faces()
    faces.append({"name": name, "encoding": encoding})
    with open(DB_PATH, "wb") as f:
        pickle.dump(faces, f)

def compare_faces(known_faces, face_encoding, tolerance=0.45):
    for face in known_faces:
        dist = np.linalg.norm(face["encoding"] - face_encoding)
        if dist < tolerance:
            return face["name"]
    return "Unknown"