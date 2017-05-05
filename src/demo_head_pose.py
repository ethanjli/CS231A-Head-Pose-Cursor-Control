import head_pose

def echo(yaw, pitch):
    print("Yaw: " + str(yaw) + ", pitch: " + str(pitch))

tracker = head_pose.HeadPose()
tracker.monitor_sync(echo)

