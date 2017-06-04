"""
demo_facial_landmarks.py
This script demonstrates calibrated facial landmarks tracking.
"""
import render
import scene_manager
import animation

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)

facial_landmarks = animation.FaceAxesAnimator()
facial_landmarks.register_rendering_pipeline(pipeline)

facial_landmarks.animate_async()
pipeline.start_rendering()

facial_landmarks.stop_animating()
