import cv2
import time
import logging
import numpy as np

LOGFILE = 'camera.log'
logging.basicConfig(filename=LOGFILE, level='DEBUG')

webcam = cv2.VideoCapture(0)
img_count = 0

while True:

    # if webcam is opened
    if not webcam.isOpened():
        logging.warning('Unable to connect to camera.')
        time.sleep(5)
        continue

    # get image from camera
    ret, frame = webcam.read()

    # if frame is read correctly ret is True
    if not ret:
        logging.warning("Can't receive frame (stream end?). Exiting ...")
        break

    # show image
    cv2.imshow('Press ESC - for quite. SPACE - for snapshot', frame)

    # key for snapshot or quit
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed for quit
        logging.info('Escape hit, closing...')
        break
    elif k % 256 == 32:
        # SPACE pressed for save
        img_name = f'opencv_frame_{img_count}.png'
        cv2.imwrite(img_name, frame)
        logging.info(f'{img_name} written!')

        # read image
        img = cv2.imread(img_name, 1)

        # get width and height for points rectangle and line
        height, width = img.shape[:2]
        # upper left point
        A1_up_left = (int(width / 4), int(height / 4))
        # downer right point
        A2_down_right = (int(3 * width / 4), int(3 * height / 4))

        # convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        colors_rectangle = (0, 255, 0)
        colors_line = (255, 0, 0)
        thickness = 5

        # draw rectangle on image
        cv2.rectangle(gray_img, A1_up_left, A2_down_right, colors_rectangle, thickness)
        # draw line on image
        cv2.line(gray_img, A1_up_left, A2_down_right, colors_line, thickness)

        cv2.imshow(img_name, gray_img)
        img_count += 1

# close everything
webcam.release()
cv2.destroyAllWindows()