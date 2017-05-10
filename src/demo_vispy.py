# !/usr/bin/env python2
"""Demonstrating a checkerboard."""
import numpy as np

import render
import visuals.canvas

pipeline = render.RenderingPipeline()
canvas = pipeline.instantiate_visual(visuals.canvas.CanvasVisual, 'checkerboard')
pipeline.start_rendering()
