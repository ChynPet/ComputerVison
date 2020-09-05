import numpy as np
import cv2
import time
import logging

LOGFILE = 'camera.log'
CODEC = 'DIVX'

logging.basicConfig(filename=LOGFILE, level='DEBUG')

webcam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*CODEC)
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480), False)
# flag to track or play video
flag_played_video = False

while True:
    # if webcam is opened
    if not webcam.isOpened():
        logging.warning('Unable to connect to camera.')
        time.sleep(5)
        continue

    # get image from camera
    res, frame = webcam.read()

    # if frame is read correctly res is True
    if not res:
        logging.warning("Can't receive frame (stream end?). Exiting ...")
        break

    # show image
    cv2.imshow('Press ESC - for quite. SPACE - for write frame to video', frame)

    # key for create a frame for a video sequence or output
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed for quit

        logging.info('Escape hit, closing...')
        break
    elif k % 256 == 32:
        # SPACE pressed for create a frame for a video sequence

        # if a video was played, to create a new object and rewrite the file for video
        if flag_played_video:
            flag_played_video = False
            out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480), False)

        # get width and height for points rectangle and line
        height, width = frame.shape[:2]
        # the magnitude of the random displacement of the square and line in the frame
        bais = np.random.randint(low=0, high=35)
        # upper left point
        A1_up_left = (int(width / 4) + bais, int(height / 4) + bais)
        # downer right point
        A2_down_right = (int(3 * width / 4) + bais, int(3 * height / 4) + bais)

        colors_rectangle = (0, 255, 0)
        colors_line = (255, 0, 0)
        thickness = 5

        # convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # draw rectangle on frame
        cv2.rectangle(gray_frame, A1_up_left, A2_down_right, colors_rectangle, thickness)
        # draw line on frame
        cv2.line(gray_frame, A1_up_left, A2_down_right, colors_line, thickness)

        cv2.imshow('photo', gray_frame)
        # write the frame to the video sequence
        out.write(gray_frame)
    elif k % 256 == 112 or k % 256 == 80:
        # P or p play video, which was written

        flag_played_video = True

        # if a video was recorded, complete it
        if out.isOpened():
            out.release()

        video = cv2.VideoCapture('output.avi')

        while video.isOpened():
            ret, frame = video.read()

            # if frame is read correctly ret is True
            if not ret:
                logging.warning("Can't read frame from 'output.avi'")
                break

            cv2.imshow('Playing video', frame)

            # exit check
            if cv2.waitKey(1) % 256 == 113:
                break

        # release capture
        video.release()


# close everything
webcam.release()
out.release()
cv2.destroyAllWindows()