from gtts import gTTS
import pygame
import time
import os

AUDIO_FILE = "alert_test.mp3"

if not os.path.exists(AUDIO_FILE):
    tts = gTTS("bahaya api terdeteksi", lang="id")
    tts.save(AUDIO_FILE)

pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)
pygame.mixer.music.play()

print("Suara sedang diputar...")
time.sleep(5)  # tunggu 5 detik biar sempat kedengeran
