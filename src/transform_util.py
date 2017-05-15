import numpy as np

MONITOR_HEIGHT = 29.8  # cm
MONITOR_WIDTH = 53.0  # cm
CAMERA_X = MONITOR_WIDTH / 2.0  # cm from the left and right edges of the screen (camera is between the post-it note tabs)
CAMERA_Y = 3.4  # cm above the top edge of the screen (the top row of pixels)
SCREEN_HEIGHT = 1080  # px
SCREEN_WIDTH = 1920  # px

#def render_xy_px_to_screen_xy_px(render_x, render_y):
#    screen_center = (SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0)
#    return (render_x + screen_center[0], render_y - screen_center[1])

#def screen_xy_px_to_render_xy_px(screen_x, screen_y):
#    screen_center = (SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0)
#    return (screen_x - screen_center[0], screen_y + screen_center[1])

#def screen_xy_px_to_screen_xy_cm(screen_x, screen_y):
#    return (screen_x * MONITOR_WIDTH / SCREEN_WIDTH - CAMERA_X,
#            screen_y * MONITOR_HEIGHT / SCREEN_HEIGHT - CAMERA_Y)

#def screen_xy_cm_to_screen_xy_px(screen_x, screen_y):
#    return ((screen_x + CAMERA_X) * SCREEN_WIDTH / MONITOR_WIDTH,
#            (screen_y + CAMERA_Y) * SCREEN_HEIGHT / MONITOR_HEIGHT)

#def render_xy_to_screen_xy(render_x, render_y):
#    return screen_xy_px_to_screen_xy_cm(
#        *render_xy_px_to_screen_xy_px(render_x, render_y))

#def screen_xy_to_render_xy(screen_x, screen_y):
#    return screen_xy_px_to_render_xy_px(
#        *screen_xy_cm_to_screen_xy_px(screen_x, screen_y))

def render_xy_px_to_screen_xy_px(render_x, render_y):
    return render_x, render_y

def screen_xy_px_to_render_xy_px(screen_x, screen_y):
    return screen_x, screen_y

def screen_xy_px_to_screen_xy_cm(screen_x, screen_y):
    return (screen_x * MONITOR_WIDTH / SCREEN_WIDTH,
            screen_y * MONITOR_HEIGHT / SCREEN_HEIGHT)

def screen_xy_cm_to_screen_xy_px(screen_x, screen_y):
    return (screen_x * SCREEN_WIDTH / MONITOR_WIDTH,
            screen_y * SCREEN_HEIGHT / MONITOR_HEIGHT)

def render_xy_to_screen_xy(render_x, render_y):
    return screen_xy_px_to_screen_xy_cm(
        *render_xy_px_to_screen_xy_px(render_x, render_y))

def screen_xy_to_render_xy(screen_x, screen_y):
    return screen_xy_px_to_render_xy_px(
        *screen_xy_cm_to_screen_xy_px(screen_x, screen_y))

class Calibration:
  # Pitch, yaw, roll in degrees
  # (x, y, z) is calibrated initial position
  def __init__(self, pitch=0, yaw=0, roll=0, x=0, y=0, z=0):
    self.pitch, self.yaw, self.roll = pitch, yaw, roll
    self.x, self.y, self.z = x, y, z

  # Transform a point (pitch, yaw, roll in degrees)
  # Returns a tuple of new screen coordinates (x', y')
  #
  # x, y, and z are distances measured from the person's head to the camera
  # x direction corresponds to moving rightwards on the screen
  # y direction corresponds to moving up (in the natural sense of up)
  # z direction corresponds to depth of camera
  # screen_x and screen_y are measured relative to the position of the camera on the screen
  #
  # all numbers are in arbitrary units, but the units measuring screen_x, screen_y, x, y, z
  # must be the same
  def transform(self, screen_x, screen_y, pitch, yaw, roll, x, y, z):
    dpitch = np.deg2rad(pitch - self.pitch)
    dyaw = np.deg2rad(yaw - self.yaw)
    droll = np.deg2rad(roll - self.roll)
    z_hat = np.array([0, 0, 1.])
    y_hat = np.array([0, 1., 0])

    # Ignore roll for now

    R_pitch = np.array([
      [1, 0, 0],
      [0, np.cos(dpitch), np.sin(dpitch)],
      [0, -np.sin(dpitch), np.cos(dpitch)]
      ])
    # CW yaw direction viewed from above
    R_yaw = np.array([
      [np.cos(dyaw), 0, np.sin(dyaw)],
      [0, 1, 0],
      [-np.sin(dyaw), 0, np.cos(dyaw)]
      ])
    # Order is roll -> pitch -> yaw
    T = R_yaw.dot(R_pitch)

    z_hat2 = T.dot(z_hat)
    y_hat2 = T.dot(y_hat)
    x_hat2 = np.cross(y_hat2, z_hat2)

    # New camera position
    #O2 = np.array([self.x, self.y, self.z]) \
    #  - z * z_hat2 - y * y_hat2 - x * x_hat2
    #O2 = np.array([x, y, z])
    O2 = np.array([x - self.x, y - self.y, z - self.z])

    p2 = np.array([screen_x, screen_y, 0.])
    p2[2] = (z_hat2.dot(O2) - z_hat2[:2].dot(p2[:2])) / z_hat2[2]

    dpos = p2 - O2
    x2 = dpos.dot(x_hat2)
    y2 = dpos.dot(y_hat2)

    return (x2, y2)

# Testing code
if __name__ == "__main__":
  c = Calibration(0, 0, 0, 2, 3, 10)
  print(c.transform(10, -10, 5, 20, -15, 0, 0, 0))
