"""
demo_head_pose.py
This script demonstrates integration of head pose tracking with gazr and the RenderingPipeline
with static point cloud data.
Pipe the stdout of gazr's gazr_estimate_head_direction binary to the stdin of this script to
run it.
"""
import render
import visuals.canvas
import animation

pipeline = render.RenderingPipeline()
canvas = pipeline.instantiate_visual(visuals.canvas.CanvasVisual, 'checkerboard')
head_pose_animator = animation.HeadPoseAnimator()
head_pose_animator.register_rendering_pipeline(pipeline)
pipeline.start_rendering()
head_pose_animator.clean_up()

