import cv2
import mediapipe as mp
import numpy as np
import pygame
from gtts import gTTS
import os

# Inisialisasi pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Inisialisasi FaceMesh
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Fungsi buat beep alarm (looping pakai pygame)
def make_beep(frequency=800, volume=0.5):
    sample_rate = 44100
    t = np.linspace(0, 1.0, int(sample_rate * 1.0), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * (32767 * volume)).astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

# Siapkan beep alarm
alarm_sound = make_beep(frequency=900, volume=0.6)

# Buat file suara Google sekali saja
if not os.path.exists("ngantuk.mp3"):
    tts = gTTS(text="Hati-hati, anda terdeteksi mengantuk", lang="id")
    tts.save("ngantuk.mp3")

# Fungsi play audio google
def play_google_voice():
    pygame.mixer.music.load("ngantuk.mp3")
    pygame.mixer.music.play()

# Fungsi EAR (Eye Aspect Ratio)
def eye_aspect_ratio(landmarks, eye_idx):
    p1 = np.array([landmarks[eye_idx[0]].x, landmarks[eye_idx[0]].y])
    p2 = np.array([landmarks[eye_idx[1]].x, landmarks[eye_idx[1]].y])
    p3 = np.array([landmarks[eye_idx[2]].x, landmarks[eye_idx[2]].y])
    p4 = np.array([landmarks[eye_idx[3]].x, landmarks[eye_idx[3]].y])
    p5 = np.array([landmarks[eye_idx[4]].x, landmarks[eye_idx[4]].y])
    p6 = np.array([landmarks[eye_idx[5]].x, landmarks[eye_idx[5]].y])

    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)

    return (vert1 + vert2) / (2.0 * horiz)

# Threshold & frame untuk deteksi
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 15

counter_ngantuk = 0
alarm_on = False
voice_played = False

cap = cv2.VideoCapture(1)  # ganti ke 1 kalau pakai kamera eksternal

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
                rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESH:
                    counter_ngantuk += 1
                    if counter_ngantuk >= EAR_CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_sound.play(-1)  # bunyi terus
                            alarm_on = True
                            voice_played = False
                        if not voice_played:
                            play_google_voice()
                            voice_played = True

                        # Teks peringatan
                        cv2.putText(frame, "HATI-HATI! TERDETEKSI MENGANTUK",
                                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3)
                else:
                    counter_ngantuk = 0
                    if alarm_on:
                        alarm_sound.stop()
                        alarm_on = False
                        voice_played = False

                # Tampilkan nilai EAR
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
