import os
import cv2
import time

def load_images_from_dir(dir):
    images = []

    for entry in os.listdir(dir):
        entrypath = os.path.join(dir, entry)
        if not os.path.isfile(entrypath):
            continue

        img = cv2.imread(entrypath)
        if img is None:
            continue

        images.append(img)

    return images

def main():
    images = load_images_from_dir("pictures")

    if len(images) == 0:
        print("No images")
        return

    index = 0
    images_len = len(images)

    is_animation_active = False
    animation_slide_time_s = 3

    last_change = time.time()

    while True:
        img = images[index]
        cv2.imshow("result", img)

        key = cv2.waitKey(30)

        if key == ord("e"):
            break

        if key == ord("a"):
            is_animation_active = not is_animation_active

        if not is_animation_active:
            if key == ord("q"):
                index = (index + 1) % images_len
            elif key == ord("w"):
                index = (index - 1) % images_len
        else:
            if key == ord("z"):
                animation_slide_time_s += 1
            elif key == ord("x"):
                animation_slide_time_s = max(1, animation_slide_time_s - 1)
            
            if time.time() - last_change >= animation_slide_time_s:
                index = (index + 1) % images_len
                last_change = time.time()

    cv2.destroyAllWindows()

main()