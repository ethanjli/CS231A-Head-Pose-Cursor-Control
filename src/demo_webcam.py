#!/usr/bin/env python2
import cv2
import dlib

import util

predictor_path = util.MODEL_PATH

window = dlib.image_window()
vc = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

framerate = util.FramerateCounter()
while rval:
    framerate.tick()
    print(framerate.query())
    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    dets = detector(image, 0)

    window.clear_overlay()
    window.set_image(image)
    print('Number of faces detected: {}'.format(len(dets)))
    for (k, d) in enumerate(dets):
        print('Detection {}: Left: {} Top: {} Right: {} Bottom: {}'.format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(image, d)
        print('Part 0: {}, Part 1: {} ...'.format(shape.part(0), shape.part(1)))
        window.add_overlay(shape)
    window.add_overlay(dets)

    rval, frame = vc.read()

