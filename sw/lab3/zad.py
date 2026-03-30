import cv2
import os

img_color = cv2.imread('dokument.jpg')
img_resized = cv2.resize(img_color, None, fx=1.5, fy=1.5)

gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imwrite('document_binary.jpg', closing)

cv2.imshow("Original", img_color)
cv2.imshow("Binary", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()

size_color = os.path.getsize('dokument.jpg')
size_binary = os.path.getsize('document_binary.jpg')

print("Rozmiar kolorowego:", size_color)
print("Rozmiar binarnego:", size_binary)