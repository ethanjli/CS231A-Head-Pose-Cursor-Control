import numpy as np

class Calibration:
  # Pitch, yaw, roll in degrees
  # (x, y, z) is calibrated initial position
  def __init__(self, pitch, yaw, roll, x, y, z):
    self.pitch, self.yaw, self.roll = pitch, yaw, roll
    self.x, self.y, self.z = x, y, z

  # Transform a point (pitch, yaw, roll in degrees)
  # Point is a 2D point in screen coordinates (origin at camera)
  # Returns a tuple of new screen coordinates (x', y')
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
