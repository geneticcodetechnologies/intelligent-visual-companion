import cv2
import requests

API_URL = "http://127.0.0.1:8000/recognize-face"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("📷 Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show webcam
    cv2.imshow("Webcam", frame)

    # Send every N frames (important!)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        _, buffer = cv2.imencode(".jpg", frame)

        files = {
            "file": ("frame.jpg", buffer.tobytes(), "image/jpeg")
        }

        try:
            response = requests.post(API_URL, files=files, timeout=3)
            print("API response:", response.json())
        except Exception as e:
            print("❌ API error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
