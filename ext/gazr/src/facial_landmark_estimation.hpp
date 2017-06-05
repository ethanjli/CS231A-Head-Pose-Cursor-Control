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
    /** Return the point corresponding to the index.
    */
    cv::Point2f coordsOf(size_t face_idx, unsigned long index) const;

    //std::vector<unsigned long> points_to_use = {0, 1, 2, 14, 15, 16, 31, 32, 33, 34, 35, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}; // 27 points
    //std::vector<unsigned long> points_to_use = {30, 36, 45, 0, 16, 21, 22, 51, 57, 48, 54, 27, 62, 66, 8}; // 15 points
    // std::vector<unsigned long> points_to_use = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67}; // 68 points
    std::vector<unsigned long> points_to_use = {8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67}; // 52 points
};

#endif // __FACIAL_LANDMARK_ESTIMATION
