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

def map_3d_model(model_3d, camera_matrices):
  """
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

class StereoModelCalibration:
  def __init__(self, camera_distance, K1, K2, model_3d):
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
    """
    self._camera_distance = camera_distance
    self._camera_matrices = np.zeros((2,3,4))
    self._camera_matrices[0] = np.concatenate([K1, np.zeros((3,1))], axis=1)
    M2 = K1.dot(np.concatenate([np.eye(3), np.array([[-camera_distance,0,0]]).T], axis=1))
    self._camera_matrices[1] = M2
    self._model_3d = model_3d

  def compute_RT(self, points):
    """
    Compute the RT matrix that, when applied to the original 3d model, yields the set of
    observations in points.

    Arguments:
      points: a N x 2 x 2 set of points corresponding to positions on images taken by the two cameras.
        second to last index corresponds to camera number.

    Returns:
      R: the rotation matrix (to be applied about the centroid of the object)
      T: the translation vector (displacement of centroid from calibrated position to final
        position)
    """
      pass

def test_run():
  """
  Method for testing class, run when program is __main__.
  """
  K1 = K2 = np.diag([0.5, 0.4, 1])
  model_3d = np.array([[1,1,1],
    [0,0,2],
    [0.4, 0.6, 0.2],
    [-0.2, 0.5, 0.5]])
  print 'Model:', model_3d
  smc = StereoModelCalibration(1, K1, K2, model_3d)
  points = map_3d_model(model_3d, smc._camera_matrices)
  print 'Mapped points:', points
  points += np.random.randn(*points.shape) * 0.05
  rec_3d = compute_3d_model(points, smc._camera_matrices)
  print 'Rec_3d:', rec_3d
  print 'Diff:', model_3d - rec_3d
  return smc

if __name__=='__main__':
  test_run()