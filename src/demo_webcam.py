#!/usr/bin/env python2
import numpy as np
import cv2
import dlib
import skvideo.io 
import skimage
import skimage.transform
import argparse

import util

parser = argparse.ArgumentParser(description='Run image stabilization algorithm.')
parser.add_argument('--use-file', type=str, default='',
                    help='a file to use instead of webcam input')
args = parser.parse_args()

predictor_path = util.MODEL_PATH

window = dlib.image_window()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# TODO: instead of using fx and fy, perhaps use dsize for resizing
if args.use_file != '':
    vc = skvideo.io.VideoCapture(args.use_file)
    use_bgr2rgb = False
    fx = 1.
    fy = 1.
else:
    vc = cv2.VideoCapture(0)
    use_bgr2rgb = True
    fx = 0.25
    fy = 0.25
if vc.isOpened():
    rval, frame = vc.read()
else:
    print("Video capture not succesfully opened!")
    rval = False

framerate = util.FramerateCounter()
while rval:
    framerate.tick()
    print(framerate.query())
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if use_bgr2rgb else frame
    image = cv2.flip(image, 1)
    small_image = cv2.resize(image, (0, 0), fx=fx, fy=fy)
    dets = detector(small_image, 0)

    window.clear_overlay()
    window.set_image(small_image)
    print('Number of faces detected: {}'.format(len(dets)))
    for (k, d) in enumerate(dets):
        print('Detection {}: Left: {} Top: {} Right: {} Bottom: {}'.format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(small_image, d)
        print('Part 0: {}, Part 1: {} ...'.format(shape.part(0), shape.part(1)))
        window.add_overlay(shape)
    window.add_overlay(dets)

    rval, frame = vc.read()

