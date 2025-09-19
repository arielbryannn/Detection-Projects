import cv2
import numpy as np
import pygame
from gtts import gTTS
import os
import time
from datetime import datetime
import smtplib
from email.message import EmailMessage

# ========================
# Setup suara
# ========================
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def make_beep(frequency=1000, volume=0.5):
    sample_rate = 44100
    t = np.linspace(0, 1.0, int(sample_rate * 1.0), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * (32767 * volume)).astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo_wave)

# Alarm beep
alarm_beep = make_beep(frequency=900, volume=0.6)

# Suara Google TTS sekali
AUDIO_FILE = "alert_api.mp3"
if not os.path.exists(AUDIO_FILE):
    tts = gTTS("bahaya api terdeteksi", lang="id")
    tts.save(AUDIO_FILE)

def play_google_voice():
    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()

# ========================
# Setup email
# ========================
SENDER_EMAIL = "arielbrayen5@gmail.com"
SENDER_PASSWORD = "savbytabfysxuiuy"  # ini App Password yang baru
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
RECEIVER_EMAIL = "arielbrayen5@gmail.com"


def send_email_with_attachment(img_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"ALERT: Api terdeteksi - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content("Bahaya: api terdeteksi. Lampiran berisi screenshot kamera.")

        with open(img_path, "rb") as f:
            data = f.read()
            msg.add_attachment(data, maintype="image", subtype="png", filename=os.path.basename(img_path))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"[EMAIL] Terkirim ke {RECEIVER_EMAIL}")
    except Exception as e:
        print("[EMAIL] Gagal:", e)

# ========================
# Setup kamera & deteksi
# ========================
MIN_FIRE_AREA = 800
cap = cv2.VideoCapture(1)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

alarm_on = False
last_alert_time = 0
COOLDOWN = 10   # jeda 10 detik antar deteksi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Warna api (oranye + merah)
    lower1 = np.array([10, 100, 100])
    upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 120, 120])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_FIRE_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            fire_detected = True

    # Cek pergerakan
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    movement = np.sum(thresh) / 255
    prev_gray = gray

    now = time.time()
    if fire_detected and movement > 1000:
        if now - last_alert_time > COOLDOWN:   # hanya sekali per 10 detik
            cv2.putText(frame, "!!! DETEKSI API !!!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

            alarm_beep.play(-1)   # looping beep
            play_google_voice()

            # Simpan screenshot
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"screenshot_fire_{ts}.png"
            cv2.imwrite(img_path, frame)
            print(f"[SS] Screenshot tersimpan: {img_path}")

            # Kirim email
            send_email_with_attachment(img_path)

            last_alert_time = now
            alarm_on = True
    else:
        if alarm_on and (now - last_alert_time > 3):  # stop beep setelah 3 detik
            alarm_beep.stop()
            alarm_on = False

    cv2.imshow("Deteksi Api", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
alarm_beep.stop()
cv2.destroyAllWindows()
