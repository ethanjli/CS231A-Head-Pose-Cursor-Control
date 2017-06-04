"""
demo_head_pose.py
This script demonstrates integration of head pose tracking with gazr and the RenderingPipeline
with a canvas.
"""
import render
import scene_manager
import animation

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_checkerboard()

head_pose_animator = animation.HeadPoseAnimator()
head_pose_animator.register_visual_node(scene_manager.checkerboard)
head_pose_animator.register_rendering_pipeline(pipeline)
head_pose_animator.animate_async()

pipeline.start_rendering()

head_pose_animator.stop_animating()
