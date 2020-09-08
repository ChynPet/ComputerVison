import numpy as np
import cv2
import time
import logging

LOGFILE = 'camera.log'
CODEC = 'DIVX'

logging.basicConfig(filename=LOGFILE, level='DEBUG')

webcam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*CODEC)
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480))
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
    cv2.imshow('Press ESC - for quite. SPACE - for write frame to video. P or p play video', frame)

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
            out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480))

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
        # canvas for draw rectangle and line
        canvas = np.zeros((height, width, 3), np.uint8)

        # convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # draw rectangle on canvas
        cv2.rectangle(canvas, A1_up_left, A2_down_right, colors_rectangle, thickness)
        # draw line on frame
        cv2.line(canvas, A1_up_left, A2_down_right, colors_line, thickness)

        # geting mask for draw color rectangle and line on gray image
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        gray_frame_bitwise = cv2.bitwise_and(gray_frame, gray_frame, mask=mask_inv)
        canvas_bitwise = cv2.bitwise_and(canvas, canvas, mask=mask)

        res = canvas_bitwise + gray_frame_bitwise[:, :, np.newaxis]
        cv2.imshow('photo', res)
        # write the frame to the video sequence
        out.write(res)
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

            cv2.imshow('Playing video. q for quit', frame)

            # exit check
            if cv2.waitKey(1) % 256 == 113:
                break

        # release capture
        video.release()


# close everything
webcam.release()
out.release()
cv2.destroyAllWindows()