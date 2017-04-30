#include <string>
#include <set>

#include "head_pose_estimation.hpp"

// opencv2
#include <opencv2/core/core.hpp>

// ROS
#include <ros/ros.h>
#include <std_msgs/Char.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>

#include <image_geometry/pinhole_camera_model.h>

#include <cv_bridge/cv_bridge.h>


class HeadPoseEstimator
{
public:

    HeadPoseEstimator(ros::NodeHandle& rosNode,
                      const std::string& prefix,
                      const std::string& modelFilename = "");

private:

    ros::NodeHandle& rosNode;
    image_transport::ImageTransport it;
    image_transport::CameraSubscriber sub;
    image_transport::Publisher pub;

    ros::Publisher nb_detected_faces_pub;

    tf::TransformBroadcaster br;
    tf::Transform transform;

    image_geometry::PinholeCameraModel cameramodel;
    cv::Mat cameraMatrix, distCoeffs;

    cv::Mat inputImage;
    HeadPoseEstimation estimator;

    // prefix prepended to TF frames generated for each frame
    std::string facePrefix;

    bool warnUncalibratedImage;

    void detectFaces(const sensor_msgs::ImageConstPtr& msg,
                     const sensor_msgs::CameraInfoConstPtr& camerainfo);
};

