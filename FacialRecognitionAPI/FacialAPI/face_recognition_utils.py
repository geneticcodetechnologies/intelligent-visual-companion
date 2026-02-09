#utilities for face recognition
import FacialAPI.FacialRecognitionAPI
import numpy as np
import cv2
from typing import List, Optional


def load_image_bytes_to_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr

def get_face_encodings_from_bgr(bgr_image):
    # face_recognition expects RGB
    # detect face locations and compute encodings
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    locations = FacialAPI.FacialRecognitionAPI.face_locations(rgb, model="hog")  # or "cnn"
    encodings = FacialAPI.FacialRecognitionAPI.face_encodings(rgb, locations)
    return locations, encodings

def encode_image_bytes(image_bytes: bytes):
    bgr = load_image_bytes_to_bgr(image_bytes)
    if bgr is None:
        return [], []
    locs, encs = get_face_encodings_from_bgr(bgr)
    return locs, encs

def compare_encodings(known_encodings: List[np.ndarray], query_encoding: np.ndarray):
    # returns (best_index, best_distance)
    if not known_encodings:
        return None, None
    dists = FacialAPI.FacialRecognitionAPI.face_distance(known_encodings, query_encoding)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    return best_idx, best_dist

