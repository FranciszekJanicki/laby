import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Blad kamery")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 500:

            x, y, w, h = cv2.boundingRect(c)

            if 20 < w < 300 and 20 < h < 300:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("mask", mask)
    cv2.imshow("tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()