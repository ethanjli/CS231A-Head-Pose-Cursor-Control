# !/usr/bin/env python2
"""Demonstrating a checkerboard."""
import numpy as np

import render
import scene_manager

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_checkerboard()
pipeline.start_rendering()
