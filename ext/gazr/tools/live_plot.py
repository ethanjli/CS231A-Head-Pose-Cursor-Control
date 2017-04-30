#! /usr/bin/env python

import sys

import time
from numpy import arange
import matplotlib.pyplot as plt

PLOT_WIDTH=100
plt.axis([0, PLOT_WIDTH, -90, 90])
plt.ion()

pitch = [0] * PLOT_WIDTH
yaw = [0] * PLOT_WIDTH
roll = [0] * PLOT_WIDTH
pitch_graph, yaw_graph, roll_graph = plt.plot(pitch, 'r', yaw, 'g', roll, 'b')

plt.legend([pitch_graph, yaw_graph, roll_graph], ['Pitch', 'Yaw', 'Roll'])
plt.show()


while True:
    line = sys.stdin.readline()
    data = eval(line)
    if "face_0" in data:

        pitch.append(data["face_0"]["pitch"]-180)
        del pitch[0]
        pitch_graph.set_data(arange(0, len(pitch)), pitch)

        yaw.append(data["face_0"]["yaw"]-180)
        del yaw[0]
        yaw_graph.set_data(arange(0, len(yaw)), yaw)

        roll.append(data["face_0"]["roll"])
        del roll[0]
        roll_graph.set_data(arange(0, len(roll)), roll)

        plt.xlim([max(0,len(pitch) - PLOT_WIDTH), max(0,len(pitch) - PLOT_WIDTH) + PLOT_WIDTH])
        plt.draw()
        plt.pause(0.05)
    else:
        pitch.append(0)
        yaw.append(0)
        roll.append(0)

