import cv2
import numpy as np

img = cv2.imread("drone_ship.jpg")
if img is None:
    raise FileNotFoundError("Brak drone_ship.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [largest], -1, (0,255,0), 3)

cv2.imshow("Edges", edges)
cv2.imshow("Ship outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()