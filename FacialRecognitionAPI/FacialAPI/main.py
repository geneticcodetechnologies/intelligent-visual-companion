# main.py
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import os
import uuid
from fastapi.middleware.cors import CORSMiddleware

import shutil
from FacialAPI.face_recognizer import get_face_encoding
from FacialAPI.database import save_face, get_all_faces
import face_recognition

from FacialAPI.db  import SessionLocal, Person
from FacialAPI.face_recognition_utils import encode_image_bytes, compare_encodings

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (OK for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# config
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
RECOGNITION_THRESHOLD = 0.55  # tune this (face_recognition typical 0.4-0.6)

@app.post("/register-face")
async def register_face(name: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    encoding = get_face_encoding(frame)
    if encoding is None:
        return {"error": "No face detected"}

    save_face(name, encoding)
    return {"status": "Face saved", "name": name}

@app.post("/recognize-face")
async def recognize_face_endpoint(file: UploadFile = File(...)):
    ...
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    encoding = get_face_encoding(frame)
    if encoding is None:
        return {"error": "No face detected"}

    known_faces = get_all_faces()
    for name, known_encoding in known_faces:
        match = face_recognition.compare_faces(
            [known_encoding], encoding, tolerance=0.5
        )

        if match[0]:
            return {"recognized": True, "name": name}

    return {"recognized": False}







