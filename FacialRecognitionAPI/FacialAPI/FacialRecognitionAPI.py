import cv2
import numpy as np

# Load Haar Cascade once (fast + offline)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face_and_encoding(frame):
    """
    Detects ONE face in the frame and returns:
    - bounding box (x, y, w, h)
    - face encoding (numpy array)

    Returns (None, None) if no face is found
    """

    if frame is None:
        return None, None

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None

    # Take the first detected face
    x, y, w, h = faces[0]

    # Crop the face region
    face_img = frame[y:y+h, x:x+w]

    if face_img.size == 0:
        return None, None

    # Normalize face image to fixed size
    face_img = cv2.resize(face_img, (100, 100))

    # Simple face encoding (flatten + normalize)
    encoding = face_img.flatten().astype("float32") / 255.0

    return (x, y, w, h), encoding

