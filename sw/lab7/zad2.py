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

saved_background = None
saved_current = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh_val = cv2.getTrackbarPos("Threshold", "foreground image")

    if saved_background is not None and saved_current is not None:

        diff = cv2.absdiff(saved_background, saved_current)

        _, foreground = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("background image", saved_background)
        cv2.imshow("current image", saved_current)
        cv2.imshow("foreground image", foreground)

    else:
        cv2.imshow("current image", gray)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('a'):
        saved_background = gray.copy()

    if key == ord('x'):
        saved_current = gray.copy()

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()