import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Blad kamery")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    foreground = fgbg.apply(gray)

    cv2.imshow("current image", gray)
    cv2.imshow("foreground image (MOG2)", foreground)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()