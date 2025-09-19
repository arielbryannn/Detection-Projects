import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import time
import os

# masukin pygame mixer
pygame.mixer.init()

# masukin mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mapping jari ke teks ucapan
jari_teks = {
    "thumb": "Halo, perkenalkan",
    "index": "Nama saya Ariel",
    "middle": "Salam kenal",
    "ring": "Terima kasih",
    "pinky": "Sampai jumpa"
}

# Pre-generate suara untuk tiap jari
audio_files = {}
for jari, teks in jari_teks.items():
    filename = f"{jari}.mp3"
    if not os.path.exists(filename):  
        tts = gTTS(text=teks, lang="id")
        tts.save(filename)
    audio_files[jari] = filename

# Cooldown biar ngga spam suara
last_spoken = {jari: 0 for jari in jari_teks.keys()}
cooldown = 3  # detik

# Fungsi buat play audio
def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

# Buka kamera (0 buat kamera bawaan laptop, isi 1 buat kamera eksternal)
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                jari_pos = {
                    "thumb": landmarks[4].x < landmarks[3].x,    # Jempol
                    "index": landmarks[8].y < landmarks[6].y,    # Telunjuk
                    "middle": landmarks[12].y < landmarks[10].y, # Tengah
                    "ring": landmarks[16].y < landmarks[14].y,   # Manis
                    "pinky": landmarks[20].y < landmarks[18].y   # Kelingking
                }

                # Cek cuman buat 1 jari ketika di angkat
                jari_aktif = [j for j, up in jari_pos.items() if up]

                if len(jari_aktif) == 1:  # hanya 1 jari aktif
                    jari = jari_aktif[0]
                    if time.time() - last_spoken[jari] > cooldown:
                        play_audio(audio_files[jari])
                        last_spoken[jari] = time.time()

        cv2.imshow("Hand Tracking with Voice", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
