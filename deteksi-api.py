import cv2
ensure_alert_audio()
cooldown = DETECTION_COOLDOWN
persistence_required = FIRE_PERSISTENCE_SECONDS
persistence_start = None


while True:
ret, frame = self.cap.read()
if not ret:
print('Kamera tidak terbuka atau frame tidak tersedia')
break


out, mask, detected, area, motion_score = self.process_frame(frame)


now = time.time()
if detected:
if persistence_start is None:
persistence_start = now
elif now - persistence_start >= persistence_required:
# detection confirmed
if now - self.last_email_time > cooldown:
self.last_email_time = now
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(SCREENSHOT_DIR, f'fire_{timestamp}.png')
cv2.imwrite(filename, frame)
print('Api terdeteksi! Menyimpan screenshot:', filename)


# play audio and send email in background threads
threading.Thread(target=play_alert_sound, daemon=True).start()
threading.Thread(target=send_email_with_attachment, args=(filename,), daemon=True).start()
else:
print('Api terdeteksi tapi masih dalam cooldown')
else:
persistence_start = None


# show windows
cv2.imshow('Camera', out)
cv2.imshow('Mask', mask)


key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
break


self.cap.release()
cv2.destroyAllWindows()




if __name__ == '__main__':
print('Memulai detektor api...')
print('MP_AVAILABLE =', MP_AVAILABLE)
detector = FireDetector(camera_index=CAMERA_INDEX)
detector.run()