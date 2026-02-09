import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

gesture = input("Enter the gesture label (e.g., A, B, C): ")
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{gesture}.csv")

cap = cv2.VideoCapture(0)

landmarks_list = []

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    print("Collecting landmarks. Press ESC to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(landmarks)

        cv2.imshow("Extract Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

if landmarks_list:
    df = pd.DataFrame(landmarks_list)
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)
    print(f"Saved {len(landmarks_list)} frames to {output_file}")
