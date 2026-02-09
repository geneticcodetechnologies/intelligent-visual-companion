import cv2
import pathlib

# API for facial detection
cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))


def detect_faces_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return [
        {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
        for (x, y, w, h) in faces
    ]


