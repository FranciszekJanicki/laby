import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Blad kamery")
    exit()

def nothing(x):
    pass

cv2.namedWindow("foreground image")
cv2.createTrackbar("Threshold", "foreground image", 25, 255, nothing)

background = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background is None:
        background = gray.copy()
    else:
        background[background < gray] += 1
        background[background > gray] -= 1

    diff = cv2.absdiff(background, gray)

    t = cv2.getTrackbarPos("Threshold", "foreground image")
    _, foreground = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

    cv2.imshow("background image", background)
    cv2.imshow("current image", gray)
    cv2.imshow("foreground image", foreground)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()