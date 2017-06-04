import head_pose

def echo(parameters):
    print(parameters)

tracker = head_pose.HeadPose()
tracker.monitor_sync(echo)

