#!/usr/bin/env python2
import numpy as np

import render

class Animator():
    def __init__(self):
        self._renderer = None

    def registerRenderer(self, renderer):
        self._renderer = renderer

    def execute(self, event):
        self._renderer.xyzrpy[5] += np.pi / 180.0
        self._renderer.update_view()


if __name__ == '__main__':
    r = render.Renderer()
    a = Animator()
    a.registerRenderer(r)
    r.startRendering(a)

