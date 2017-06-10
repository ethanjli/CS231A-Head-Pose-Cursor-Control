"""
demo_stabilization.py
This script demonstrates calibrated head pose tracking with gazr.
"""
import render
import scene_manager
import animation

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_checkerboard()

stabilizer = animation.ScreenStabilizer()
stabilizer.register_visual_node(scene_manager.checkerboard)
stabilizer.register_head_visual_node(scene_manager.axes)
stabilizer.register_rendering_pipeline(pipeline)

pipeline.start_rendering()

stabilizer.stop_animating()
