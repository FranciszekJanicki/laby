import cv2
import numpy as np

img = cv2.imread("coins.jpg")
if img is None:
    raise FileNotFoundError("Brak coins.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 150,
                           param1=120, param2=45,
                           minRadius=30, maxRadius=80)

total = 0.0

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    filtered = []
    for (x, y, r) in circles:
        keep = True
        for (fx, fy, fr) in filtered:
            dist = np.sqrt((int(x) - int(fx))**2 + (int(y) - int(fy))**2)
            if dist < max(r, fr) * 0.7:
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    radii = [r for (_,_,r) in filtered]
    threshold = (min(radii) + max(radii)) / 2 if radii else 0

    for (x, y, r) in filtered:
        if r < threshold:
            value = 0.10
        else:
            value = 1.00

        total += value

        cv2.circle(img, (x,y), r, (0,255,0), 2)
        cv2.putText(img, f"{value:.2f}", (x-20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

print(f"Suma monet: {total:.2f} PLN")

cv2.imshow("Coins", img)
cv2.waitKey(0)
cv2.destroyAllWindows()