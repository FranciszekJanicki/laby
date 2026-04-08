import cv2
import numpy as np

img = cv2.imread("fruit.jpg")
if img is None:
    raise FileNotFoundError("Brak fruit.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

def draw_objects(mask, color, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 500:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

draw_objects(mask_orange, (0,140,255), "Orange")
draw_objects(mask_green, (0,255,0), "Apple")

cv2.imshow("Fruits detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()