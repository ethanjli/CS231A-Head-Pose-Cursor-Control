import facial_landmarks
import animation

class CSVLogger():
    def __init__(self):
        header = 't (sample #)'
        for i in range(facial_landmarks.NUM_KEYPOINTS):
            header += ', x' + str(i) + ' (px), y' + str(i) + ' (px)'
        print header
        self.t = 0

    def echo(self, parameters):
        row = str(self.t)
        for i in range(facial_landmarks.NUM_KEYPOINTS):
            row += ', ' + str(parameters[i,0]) + ', ' + str(parameters[i,1])
        print row
        self.t += 1

tracker = facial_landmarks.FacialLandmarks(filters=animation.make_facial_raw_filters())
logger = CSVLogger()
tracker.monitor_sync(logger.echo)
