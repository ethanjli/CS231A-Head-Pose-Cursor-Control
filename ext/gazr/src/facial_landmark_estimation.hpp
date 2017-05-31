#ifndef __FACIAL_LANDMARK_ESTIMATION
#define __FACIAL_LANDMARK_ESTIMATION

#include <opencv2/core/core.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <vector>
#include <array>
#include <string>


static const int MAX_FEATURES_TO_TRACK=100;

// Interesting facial features with their landmark index
enum FACIAL_FEATURE {
    NOSE=30,
    RIGHT_EYE=36,
    LEFT_EYE=45,
    RIGHT_SIDE=0,
    LEFT_SIDE=16,
    EYEBROW_RIGHT=21,
    EYEBROW_LEFT=22,
    MOUTH_UP=51,
    MOUTH_DOWN=57,
    MOUTH_RIGHT=48,
    MOUTH_LEFT=54,
    SELLION=27,
    MOUTH_CENTER_TOP=62,
    MOUTH_CENTER_BOTTOM=66,
    MENTON=8
};

typedef std::vector<cv::Point2f> landmarks_t;


class FacialLandmarkEstimation {

public:

    FacialLandmarkEstimation(const std::string& face_detection_model = "shape_predictor_68_face_landmarks.dat", float focalLength=455.);

    void update(cv::InputArray image);

    landmarks_t landmarks(size_t face_idx) const;

    std::vector<landmarks_t> all_landmarks() const;

    float focalLength;
    float opticalCenterX;
    float opticalCenterY;

#ifdef FACIAL_LANDMARK_ESTIMATION_DEBUG
    mutable cv::Mat _debug;
#endif

private:

    dlib::cv_image<dlib::bgr_pixel> current_image;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;

    std::vector<dlib::rectangle> faces;

    std::vector<dlib::full_object_detection> shapes;


    /** Return the point corresponding to the dictionary marker.
    */
    cv::Point2f coordsOf(size_t face_idx, FACIAL_FEATURE feature) const;

};

#endif // __FACIAL_LANDMARK_ESTIMATION
