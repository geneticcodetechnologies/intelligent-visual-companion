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
async def recognize_face(file: UploadFile = File(...)):
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

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register")
async def register_person(name: str, surname: str = "", file: UploadFile = File(...)):
    image_bytes = await file.read()
    locs, encs = encode_image_bytes(image_bytes)
    if not encs:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # save image for auditing (optional)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # store with embeddings (we will store all faces found — typically one)
    encodings_serializable = [enc.tolist() for enc in encs]

    db = next(get_db())
    person = Person(name=name, surname=surname)
    person.set_embeddings(encodings_serializable)
    person.image_path = filepath
    db.add(person)
    db.commit()
    db.refresh(person)

    return {"id": person.id, "name": name, "surname": surname, "faces_registered": len(encs)}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image_bytes = await file.read()
    locs, encs = encode_image_bytes(image_bytes)
    if not encs:
        return {"matches": [], "message": "No face detected"}

    db = next(get_db())
    people = db.query(Person).all()
    results = []
    # precompute known embeddings as numpy arrays for speed
    known_embeddings = []
    known_person_map = []
    for p in people:
        embeds = p.get_embeddings()  # list of encodings
        for e in embeds:
            known_embeddings.append(np.array(e))
            known_person_map.append({"id": p.id, "name": p.name, "surname": p.surname})

    for enc in encs:
        best_idx, best_dist = compare_encodings(known_embeddings, enc)
        if best_idx is None:
            results.append({"match": None, "distance": None})
            continue
        match_person = known_person_map[best_idx]
        matched = (best_dist <= RECOGNITION_THRESHOLD)
        results.append({
            "match": match_person if matched else None,
            "distance": best_dist,
            "matched": matched
        })

    return {"matches": results}

@app.get("/people")
def list_people():
    db = next(get_db())
    people = db.query(Person).all()
    return [{"id": p.id, "name": p.name, "surname": p.surname, "image_path": p.image_path} for p in people]

@app.delete("/people/{person_id}")
def delete_person(person_id: int):
    db = next(get_db())
    p = db.query(Person).filter(Person.id == person_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Person not found")
    # optionally delete stored image
    if p.image_path and os.path.exists(p.image_path):
        try:
            os.remove(p.image_path)
        except:
            pass
    db.delete(p)
    db.commit()
    return {"deleted": person_id}



