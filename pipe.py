import cv2
import mediapipe as mp
import math

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
Gesture = ""

def y(landmarks, id): return landmarks[id].y
def x(landmarks, id): return landmarks[id].x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gesture = ""

    if results.multi_hand_landmarks:


        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),  # Titik hitam
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)               # Garis kuning
            )

            index_tip = hand_landmarks.landmark[8]   # Telunjuk
            middle_tip = hand_landmarks.landmark[12] # Tengah
            ring_tip = hand_landmarks.landmark[16]   # Manis
            pinky_tip = hand_landmarks.landmark[20]  # Kelingking

                # Hitung jarak antar ujung jari (2D Euclidean)
            def calc_distance(a, b):
                    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

            dist_index_middle = calc_distance(index_tip, middle_tip)
            dist_middle_ring = calc_distance(middle_tip, ring_tip)
            dist_ring_pinky = calc_distance(ring_tip, pinky_tip)

            lm = hand_landmarks.landmark

            # A
            all_folded = all(
                y(lm, tip) > y(lm, pip)
                for tip, pip in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            # Jempol di samping (x lebih kecil dari telunjuk)
            thumb_side = x(lm, mp_hands.HandLandmark.THUMB_TIP) < x(lm, mp_hands.HandLandmark.INDEX_FINGER_MCP)

            if all_folded and thumb_side:
                gesture = "A"

            # B
            all_fingers_up = all(
                y(lm, tip) < y(lm, pip)
                for tip, pip in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )

            #thumb_front = y(lm, mp_hands.HandLandmark.THUMB_TIP) > y(lm, mp_hands.HandLandmark.INDEX_FINGER_MCP)

            if (all_fingers_up and 
                dist_index_middle < 0.08 and 
                dist_middle_ring < 0.08 and 
                dist_ring_pinky < 0.05):
                gesture = "B"

            # C
            curve_shape = all(
                0.05 < abs(y(lm, tip) - y(lm, mcp)) < 0.15
                for tip, mcp in [
                    (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP),
                ]
            )
            # Jempol melengkung ke depan (x lebih dekat ke tengah)
            thumb_forward = x(lm, mp_hands.HandLandmark.THUMB_TIP) < x(lm, mp_hands.HandLandmark.INDEX_FINGER_MCP)

            if curve_shape and thumb_forward:
                gesture = "C"

            # D
            index_up = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP) < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            other_folded = all(
                y(lm, tip) > y(lm, pip)
                for tip, pip in [
                    (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                    (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                    (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
                ]
            )
            thumb_close = (
                abs(x(lm, mp_hands.HandLandmark.THUMB_TIP) - x(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)) < 0.05
            )

            if index_up and other_folded and thumb_close:
                gesture = "D"
           
    # Tampilkan teks di layar
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)  # Warna teks 

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
