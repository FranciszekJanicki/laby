import cv2
import numpy as np

img = cv2.imread("not_bad.jpg")

# skalowanie
scale = 800 / img.shape[1]
img_resized = cv2.resize(img, None, fx=scale, fy=scale)

cv2.imshow("oryginal", img_resized)
cv2.waitKey(0)

gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# progowanie
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("progowanie", thresh)
cv2.waitKey(0)

kernel = np.ones((3,3), np.uint8)

thresh_clean = cv2.erode(thresh, kernel, iterations=1)

cv2.imshow("po erozji", thresh_clean)
cv2.waitKey(0)

# kontury
contours, hierarchy = cv2.findContours(
    thresh_clean,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

img_contours = img_resized.copy()
cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)

cv2.imshow("kontury", img_contours)
cv2.waitKey(0)

# centroidy
centroids = []

for cnt in contours:
    M = cv2.moments(cnt)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))
        
        cv2.circle(img_contours, (cx, cy), 5, (0,0,255), -1)

cv2.imshow("centroidy", img_contours)
cv2.waitKey(0)

# transformacja perspektywiczna
pts_src = np.array(centroids, dtype="float32")

# docelowe punkty przykladowe
pts_dst = np.array([
    [0,0],
    [300,0],
    [300,300],
    [0,300]
], dtype="float32")

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img_resized, M, (300,300))

cv2.imshow("po transformacji", warped)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# dopasowanie wzorca (szablon)
template = img_gray[50:150, 50:150]

result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.7
locations = np.where(result >= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(img_resized, pt,
                  (pt[0]+template.shape[1], pt[1]+template.shape[0]),
                  (255,0,0), 2)

cv2.imshow("dopasowanie", img_resized)
cv2.waitKey(0)