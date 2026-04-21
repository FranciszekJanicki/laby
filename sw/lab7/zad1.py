import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nie mozna otworzyc kamery")
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

    diff = cv2.absdiff(background, gray)

    thresh_val = cv2.getTrackbarPos("Threshold", "foreground image")
    _, foreground = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("background image", background)
    cv2.imshow("current image", gray)
    cv2.imshow("foreground image", foreground)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('a'):
        background = gray.copy()
        print("Zapisano nowe tlo")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()