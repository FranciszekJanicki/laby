import cv2
import numpy as np

img = cv2.imread('dokument.jpg', 0)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

erode = cv2.erode(thresh, kernel, iterations=1)
dilate = cv2.dilate(thresh, kernel, iterations=1)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Threshold", thresh)
cv2.imshow("Erode", erode)
cv2.imshow("Dilate", dilate)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()