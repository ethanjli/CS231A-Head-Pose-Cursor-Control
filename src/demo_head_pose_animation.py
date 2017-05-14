"""
demo_head_pose.py
This script demonstrates integration of head pose tracking with gazr and the RenderingPipeline
with a canvas.
"""
import render
import visuals.canvas
import animation

pipeline = render.RenderingPipeline()
canvas = pipeline.instantiate_visual(visuals.canvas.CheckerboardVisual, 'checkerboard')
head_pose_animator = animation.HeadPoseAnimator()
head_pose_animator.register_visual_node(canvas)
head_pose_animator.register_rendering_pipeline(pipeline)
head_pose_animator.animate_async()
pipeline.start_rendering()
head_pose_animator.clean_up()

