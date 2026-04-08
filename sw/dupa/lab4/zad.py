import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas = np.zeros((500, 500, 3), dtype=np.uint8)

def draw_shapes(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(canvas, (x-25, y-25), (x+25, y+25), (0,255,0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(canvas, (x, y), 25, (0,0,255), -1)

cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", draw_shapes)

img = cv2.imread("road.jpg")
if img is None:
    raise FileNotFoundError("Nie znaleziono obrazu road.jpg")

rotated_img = img.copy()
angle = 0

def on_trackbar(val):
    global rotated_img, angle
    angle = val
    rows, cols = img.shape[:2]
    center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w = int((rows*sin) + (cols*cos))
    new_h = int((rows*cos) + (cols*sin))
    M[0,2] += (new_w/2) - center[0]
    M[1,2] += (new_h/2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))

cv2.namedWindow("Rotate")
cv2.createTrackbar("Angle", "Rotate", 0, 360, on_trackbar)

points_src = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_src) < 4:
        points_src.append([x,y])
        cv2.circle(img, (x,y), 5, (255,0,0), -1)

cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", select_points)

def show_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0,256])
    plt.figure()
    plt.title("Histogram Gray")
    plt.plot(hist_gray)
    plt.show()
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return equalized

points_threshold = []

def select_two_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_threshold) < 2:
        points_threshold.append((x,y))
        cv2.circle(param, (x,y), 3, (0,255,0), -1)

points_dst = []

while True:
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Rotate", rotated_img)
    cv2.imshow("Select Points", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('h'):
        show_histogram(img)
    elif key == ord('t'):
        print("Kliknij 2 punkty dla progowania kanal zielonego")
        cv2.setMouseCallback("Select Points", select_two_points, img)
        while len(points_threshold) < 2:
            cv2.imshow("Select Points", img)
            cv2.waitKey(1)
        x1,y1 = points_threshold[0]
        x2,y2 = points_threshold[1]
        green = img[y1:y2, x1:x2, 1]
        _, thresh = cv2.threshold(green, 128, 255, cv2.THRESH_BINARY)
        img[y1:y2, x1:x2, 1] = thresh
        points_threshold = []

    elif key == ord('p') and len(points_src) == 4:
        pts1 = np.float32(points_src)
        pts2 = np.float32([[0,0],[400,0],[400,300],[0,300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (400,300))
        cv2.imshow("Warped", warped)

cv2.destroyAllWindows()