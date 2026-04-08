import cv2


def main():
    cap = cv2.VideoCapture(0)

    key = ord('a')
    while key != ord('q'):
        ret, frame = cap.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        cv2.imshow('result', img_edges)
        key = cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()