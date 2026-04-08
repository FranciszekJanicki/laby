import cv2
import numpy as np

def nothing(x):
    pass

img1 = cv2.imread('lenna_noise.bmp')
img2 = cv2.imread('lenna_salt_and_pepper.bmp')

current_img = img1.copy()

cv2.namedWindow('Original')
cv2.createTrackbar('n', 'Original', 1, 10, nothing)

while True:
    n = cv2.getTrackbarPos('n', 'Original')
    k = 2 * n + 1

    avg = cv2.blur(current_img, (k, k))
    gauss = cv2.GaussianBlur(current_img, (k, k), 0)
    median = cv2.medianBlur(current_img, k)

    cv2.imshow('Original', current_img)
    cv2.imshow('Average', avg)
    cv2.imshow('Gaussian', gauss)
    cv2.imshow('Median', median)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        current_img = img1.copy()
    elif key == ord('2'):
        current_img = img2.copy()

cv2.destroyAllWindows()