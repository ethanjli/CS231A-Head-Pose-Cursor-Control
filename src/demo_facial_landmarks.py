"""
demo_facial_landmarks.py
This script demonstrates calibrated facial landmarks tracking.
"""
import facial_landmarks
import render
import scene_manager
import animation

VIEW_PRESETS = scene_manager.VIEW_PRESETS

pipeline = render.RenderingPipeline(VIEW_PRESETS['1']['camera'])
scene_manager = scene_manager.SceneManager(VIEW_PRESETS)
scene_manager.register_rendering_pipeline(pipeline)
scene_manager.add_face()
scene_manager.face_point_cloud.initialize_data(facial_landmarks.NUM_KEYPOINTS)

facial_landmarks = animation.FacePointsAnimator()
facial_landmarks.register_rendering_pipeline(pipeline)
facial_landmarks.register_visual_node(scene_manager.face_point_cloud)

facial_landmarks.animate_async()
pipeline.start_rendering()

facial_landmarks.stop_animating()
