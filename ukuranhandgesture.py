import cv2
import mediapipe as mp
import math

# Inisialisasi MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Buka kamera
cap = cv2.VideoCapture(0)

# Gunakan MediaPipe Hands
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal menangkap frame")
            break

        # Konversi ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Kembalikan ke BGR untuk ditampilkan
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ambil titik ujung jari
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

                # Tampilkan di layar
                cv2.putText(image, f"Index-Middle: {dist_index_middle:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(image, f"Middle-Ring : {dist_middle_ring:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(image, f"Ring-Pinky  : {dist_ring_pinky:.3f}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Gambar tangan
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Finger Spacing (Jarak Jari)', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
