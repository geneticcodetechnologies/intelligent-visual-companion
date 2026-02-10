import cv2
import time
import requests

from FacialAPI.FacialRecognitionAPI import detect_face_and_encoding

API_URL = "http://127.0.0.1:8000/recognize-face"

camera = cv2.VideoCapture(0)
print("   Press S to send to API")
print("   Press Q to quit")

last_api_call = 0
API_COOLDOWN = 1.0  # seconds
recognized_name = "Scanning..."

while True:
    ret, frame = camera.read()
    if not ret:
        break

    box, encoding = detect_face_and_encoding(frame)

    # If a face is detected
    if encoding is not None and box is not None:
        current_time = time.time()

        # Only call API every X seconds
        if current_time - last_api_call > API_COOLDOWN:
            _, buffer = cv2.imencode(".jpg", frame)

            files = {
                "file": ("frame.jpg", buffer.tobytes(), "image/jpeg")
            }

            try:
                response = requests.post(API_URL, files=files, timeout=2)
                data = response.json()

                if data.get("recognized"):
                    recognized_name = data.get("name", "Unknown")
                else:
                    recognized_name = "Unknown"

            except Exception as e:
                recognized_name = "API Error"
                print("❌ API error:", e)

            last_api_call = current_time

        # Draw bounding box + name
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            recognized_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
