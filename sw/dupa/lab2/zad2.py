import cv2 as cv
import os
import numpy as np

def load_image_from_dir(dir, filename):
    entrypath = os.path.join(dir, filename)
    if not os.path.isfile(entrypath):
        return None

    return cv.imread(entrypath)


def nothing(x):
    pass


img = load_image_from_dir('pictures', 'pic1.jpg')
assert img is not None, "file could not be read"

h, w, _ = img.shape
h2 = h // 2
w2 = w // 2

cv.namedWindow("trackbars")

cv.createTrackbar("gray", "trackbars", 127, 255, nothing)
cv.createTrackbar("red", "trackbars", 127, 255, nothing)
cv.createTrackbar("green", "trackbars", 127, 255, nothing)
cv.createTrackbar("blue", "trackbars", 127, 255, nothing)

while True:

    grayT = cv.getTrackbarPos("gray", "trackbars")
    redT = cv.getTrackbarPos("red", "trackbars")
    greenT = cv.getTrackbarPos("green", "trackbars")
    blueT = cv.getTrackbarPos("blue", "trackbars")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    b, g, r = cv.split(img)

    # ROI
    roi_gray = gray[0:h2, 0:w2]
    roi_red = r[0:h2, w2:w]
    roi_green = g[h2:h, 0:w2]
    roi_blue = b[h2:h, w2:w]

    _, t_gray = cv.threshold(roi_gray, grayT, 255, cv.THRESH_BINARY)
    _, t_red = cv.threshold(roi_red, redT, 255, cv.THRESH_BINARY)
    _, t_green = cv.threshold(roi_green, greenT, 255, cv.THRESH_BINARY)
    _, t_blue = cv.threshold(roi_blue, blueT, 255, cv.THRESH_BINARY)

    top = np.hstack((roi_gray, roi_red))
    bottom = np.hstack((roi_green, roi_blue))
    preview = np.vstack((top, bottom))

    topT = np.hstack((t_gray, t_red))
    bottomT = np.hstack((t_green, t_blue))
    thresh = np.vstack((topT, bottomT))

    result = img.copy()

    result[0:h2,0:w2][t_gray == 0] = 0
    result[0:h2,w2:w,2][t_red == 0] = 0
    result[h2:h,0:w2,1][t_green == 0] = 0
    result[h2:h,w2:w,0][t_blue == 0] = 0

    cv.imshow("preview", preview)
    cv.imshow("threshold", thresh)
    cv.imshow("result", result)

    if cv.waitKey(30) & 0xFF == 27:
        break

cv.destroyAllWindows()