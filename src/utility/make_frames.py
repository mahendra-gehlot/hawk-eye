import cv2
import os

cam = cv2.VideoCapture('data/videos/videoplayback.mp4')

# frame
currentframe = 0

while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        ll = len(str(currentframe))
        zeros = 5 - ll

        name = 'data/video_frames/' + ('0'*zeros) + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()

