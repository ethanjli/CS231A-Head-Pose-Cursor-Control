"""
demo_stabilization.py
This script demonstrates calibration using head pose tracking with gazr.
"""
import render
import visuals.canvas
import animation

pipeline = render.RenderingPipeline()
canvas = pipeline.instantiate_visual(visuals.canvas.CheckerboardVisual, 'checkerboard')
stabilizer = animation.ScreenStabilizer()
stabilizer.register_visual_node(canvas)
stabilizer.register_head_visual_node(pipeline.axes)
stabilizer.register_rendering_pipeline(pipeline)
pipeline.start_rendering()
stabilizer.clean_up()
