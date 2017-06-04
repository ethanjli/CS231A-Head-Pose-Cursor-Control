import numpy as np

"""
FUNCTIONS TAKEN FROM ASSIGNMENT 2
"""
def linear_estimate_3d_point(image_points, camera_matrices):
    A = []
    for point, M in zip(image_points, camera_matrices):
        A.append(point[0] * M[2:3,:] - M[0:1,:])
        A.append(point[1] * M[2:3,:] - M[1:2,:])
    A = np.concatenate(A, axis=0)
    U, S, V = np.linalg.svd(A)
    P = V[-1]
    P = P[:-1] / P[-1]
    return P

def reprojection_error(point_3d, image_points, camera_matrices):
    points = []
    P = np.append(point_3d, 1)
    for M in camera_matrices:
        y = M.dot(P)
        p = 1. / y[2] * y[0:2]
        points.append(p[None,:])
    points = np.concatenate(points, axis=0)
    error = points - image_points
    error = np.reshape(error, (2 * len(camera_matrices)))
    return error

def jacobian(point_3d, camera_matrices):
    jacobian = []
    P = np.append(point_3d, 1)
    for M in camera_matrices:
        p = M.dot(P)
        jacobian.append(((p[2] * M[0,:-1] - p[0] * M[2,:-1]) / p[2] ** 2)[None,:])
        jacobian.append(((p[2] * M[1,:-1] - p[1] * M[2,:-1]) / p[2] ** 2)[None,:])
    jacobian = np.concatenate(jacobian, axis=0)
    return jacobian

def nonlinear_estimate_3d_point(image_points, camera_matrices):
    num_iters = 0
    P_hat = linear_estimate_3d_point(image_points, camera_matrices)
    for iter_num in xrange(num_iters):
        J = jacobian(P_hat, camera_matrices)
        e = reprojection_error(P_hat, image_points, camera_matrices)
        P_hat -= np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)
    return P_hat
"""
END FUNCTIONS TAKEN FROM ASSIGNMENT 2
"""

def compute_3d_model(points, camera_matrices):
  """
  Compute the set of 3d points corresponding to the paired observations.

  Arguments:
    points: a N x 2 x 2 set of points corresponding to positions on images taken by the two cameras.
      second to last index corresponds to camera number.
    camera_matrices: a 2 x 3 x 4 matrix containing the camera matrices M1 and M2

  Returns:
    points_3d: a N x 3 matrix of the triangulated points
  """
  points_3d = []
  for point_pair in points:
    point_3d = nonlinear_estimate_3d_point(point_pair, camera_matrices)
    points_3d.append(point_3d)
  points_3d = np.array(points_3d)
  return points_3d

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# Output is in degrees.
def rotationMatrixToEulerAngles(R):
  sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
   
  singular = sy < 1e-6

  if not singular :
    x = np.arctan2(R[2,1] , R[2,2])
    y = np.arctan2(-R[2,0], sy)
    z = np.arctan2(R[1,0], R[0,0])
  else :
    x = np.arctan2(-R[1,2], R[1,1])
    y = np.arctan2(-R[2,0], sy)
    z = 0

  return np.array([x, y, z]) * 180. / np.pi

def map_3d_model(model_3d, camera_matrices):
  """
  Map the 3d points from model_3d into 2 images using the provided camera_matrices.

  Arguments:
    model_3d: a N x 3 set of points in the 3d model
    camera_matrices: a 2 x 3 x 4 matrix containing the camera matrices M1 and M2

  Returns:
    points: a N x 2 x 2 set of points corresponding to positions on images taken by the two cameras.
      second to last index corresponds to camera number.
  """
  points = []
  for point_3d in model_3d:
    point_3d = np.append(point_3d, 1)
    point_pair = np.zeros((2,2))
    temp = camera_matrices[0].dot(point_3d)
    point_pair[0,:] = temp[0:2] / temp[2]
    temp = camera_matrices[1].dot(point_3d)
    point_pair[1,:] = temp[0:2] / temp[2]
    points.append(point_pair)
  points = np.array(points)
  return points

def make_parallel_camera_matrices(K1, K2, camera_distance):
  """
  Make the full camera matrices (with extrinsic parameters) for the parallel cameras with intrinsic
  parameters given by K1 and K2. Origin is set to be at the location of the first camera.

  Arguments:
    K1: intrinsic camera matrix of first camera
    K2: intrinsic camera matrix of second camera
    camera_distance: the lateral distance between the two cameras, assumed to be parallel
      and at the same height. positive camera distance means that from camera1's perspective,
      camera2 is on its right. the x axis is assumed to correspond to this rightward direction.

  Returns:
    camera_matrices: a 2 x 3 x 4 matrix containing the "M" camera matrices of the two cameras,
      with the first index corresponding to camera number

  We assume that moving rightward from the camera's point of view is +x, moving downward is +y,
  and moving away form the camera is +z.
  """
  camera_matrices = np.zeros((2,3,4))
  camera_matrices[0] = np.concatenate([K1, np.zeros((3,1))], axis=1)
  M2 = K1.dot(np.concatenate([np.eye(3), np.array([[-camera_distance,0,0]]).T], axis=1))
  camera_matrices[1] = M2  
  return camera_matrices

class NoIntersectionException(Exception):
  pass

class StereoModelCalibration:
  def __init__(self, camera_distance, K1, K2, model_3d=None, initial_pos=None):
    """
    Initialize the stereo model calibration. Requires two cameras with known camera matrices
    at the same height and parallel to one another.

    Arguments:
      camera_distance: the lateral distance between the two cameras, assumed to be parallel
        and at the same height. positive camera distance means that from camera1's perspective,
        camera2 is on its right. the x axis is assumed to correspond to this rightward direction.
      camera_matrices: a 2 x 3 x 3 matrix containing the camera matrices (K) of the two cameras
      model_3d: an N x 3 set of points corresponding to the positions of all keypoints in the
        reference position
      initial_pos: position on the screen (in same units as camera matrices) that the user is
        initially looking at. measured relative to the camera position.

    We assume that moving rightward from the camera's point of view is +x, moving downward is +y,
    and moving away form the camera is +z.
    """
    self._camera_matrices = make_parallel_camera_matrices(K1, K2, camera_distance)
    self._model_3d = model_3d
    self._initial_pos = initial_pos

  def compute_RT(self, points):
    """
    Compute the RT matrix that, when applied to the original 3d model, yields the set of
    observations in points.

    Arguments:
      points: a N x 2 x 2 set of points corresponding to positions on images taken by the two cameras.
        second to last index corresponds to camera number.

    Returns:
      R: the rotation matrix (to be applied about the centroid of the object) that changes the 3d
        model to the observed points
      T: the translation vector (displacement of centroid from calibrated position to final
        position)
    """
    points_3d = compute_3d_model(points, self._camera_matrices)
    centroid_ob = np.mean(points_3d, axis=0)
    centroid = np.mean(self._model_3d, axis=0)
    H = (points_3d - centroid_ob).T.dot(self._model_3d - centroid)
    U, s, V = np.linalg.svd(H)
    R = V.T.dot(U.T)
    T = centroid_ob - centroid
    return R.T, T

  def compute_gaze_location(self, points):
    """
    Compute the location that a user is looking at given a set of keypoints.

    Arguments:
      points: a N x 2 x 2 set of points corresponding to positions on images taken by the two cameras.
        second to last index corresponds to camera number.

    Returns:
      gaze_point: a 2 long vector containing the location on the screen the user is looking at,
        in the same units as camera matricess and measured relative to position of camera. uses same
        +x and +y axis as described in __init__.
    """
    R, T = self.compute_RT(points)
    centroid = np.mean(self._model_3d, axis=0)
    base_gaze_dir = np.append(self._initial_pos, 0) - centroid
    gaze_dir = R.dot(base_gaze_dir)
    new_centroid = centroid + T
    theta = -new_centroid[2] / gaze_dir[2]
    if theta < 0:
      raise NoIntersectionException("Gaze direction does not intersect with screen plane.")
    intersection = new_centroid + gaze_dir * theta
    gaze_point = intersection[0:2]
    return gaze_point

def test_run():
  """
  Method for testing functions in file, run when program is __main__.
  """
  # Test map_3d_model and compute_3d_model
  K1 = K2 = np.diag([0.5, 0.4, 1])
  model_3d = np.array([[1,1,1],
    [0,0,2],
    [0.4, 0.6, 0.2],
    [-0.2, 0.5, 0.5]])
  model_3d = np.array([[1,1,2],
    [-1,-1,2],
    [0.4, 0.6, 1.5],
    [-0.4, -0.6, 1.5],
    [0.8, -0.3, 1],
    [-0.8, 0.3, 1]])
  print 'Model:', model_3d
  smc = StereoModelCalibration(1, K1, K2, model_3d, np.array([0, 0]))
  points = map_3d_model(model_3d, smc._camera_matrices)
  print 'Mapped points:', points
  points += np.random.randn(*points.shape) * 0.05
  rec_3d = compute_3d_model(points, smc._camera_matrices)
  print 'Rec_3d:', rec_3d
  print 'Diff:', model_3d - rec_3d

  # Test compute_RT
  model_centroid = np.mean(model_3d, axis=0)
  shifted_model = model_3d - model_centroid
  angle = 0.4
  R = np.array([[np.cos(angle), np.sin(angle), 0],
    [-np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
    ])
  T = np.array([1, 2, 3])
  shifted_model = shifted_model.dot(R.T)
  shifted_model += model_centroid
  shifted_model += T
  points = map_3d_model(shifted_model, smc._camera_matrices)
  points += np.random.randn(*points.shape) * 0.00 # Add noise if desired
  R_com, T_com = smc.compute_RT(points)
  print 'R_act', R
  print 'T_act', T
  print 'R_com', R_com
  print 'T_com', T_com
  print 'act euler angles:', rotationMatrixToEulerAngles(R)
  print 'com euler angles:', rotationMatrixToEulerAngles(R_com)

  # Test compute_gaze_location
  point = smc.compute_gaze_location(points)
  print 'gaze point:', point
  return smc

if __name__=='__main__':
  test_run()