import cv2
from FacialAPI.FacialRecognitionAPI import detect_face_and_encoding
from FacialAPI.face_recognition import load_faces, compare_faces

camera = cv2.VideoCapture(0)

print("🔴 Press Q to quit")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    box, encoding = detect_face_and_encoding(frame)

    if encoding is not None:
        known_faces = load_faces()
        name = compare_faces(known_faces, encoding)

        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()