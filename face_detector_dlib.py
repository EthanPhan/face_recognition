#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: _.py
File description
"""


__author__ = 'Ethan'


# import required packages
import cv2
import dlib
import argparse

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()
# load input image
image = cv2.imread(args.image)
if image is None:
    print("Could not read input image")
    exit()


# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)


# apply face detection (cnn)
faces = cnn_face_detector(image, 1)

# loop over detected faces
for face in faces:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
    # draw box over face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

'''
# write at the top left corner of the image
# for color identification
img_height, img_width = image.shape[:2]
cv2.putText(image, "CNN", (img_width-50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 2)
'''

# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()
# save output image
cv2.imwrite("cnn_face_detection.png", image)
# close all windows
cv2.destroyAllWindows()
