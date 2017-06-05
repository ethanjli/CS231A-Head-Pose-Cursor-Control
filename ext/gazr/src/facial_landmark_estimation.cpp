#include <cmath>
#include <ctime>

#include <opencv2/calib3d/calib3d.hpp>

#ifdef FACIAL_LANDMARK_ESTIMATION_DEBUG
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#endif

#include "facial_landmark_estimation.hpp"

using namespace dlib;
using namespace std;
using namespace cv;

inline Point2f toCv(const dlib::point& p)
{
    return Point2f(p.x(), p.y());
}


FacialLandmarkEstimation::FacialLandmarkEstimation(const string& face_detection_model, float focalLength) :
        focalLength(focalLength),
        opticalCenterX(-1),
        opticalCenterY(-1)
{

        // Load face detection and pose estimation models.
        detector = get_frontal_face_detector();
        deserialize(face_detection_model) >> pose_model;

}


void FacialLandmarkEstimation::update(cv::InputArray _image)
{

    Mat image = _image.getMat();

    if (opticalCenterX == -1) // not initialized yet
    {
        opticalCenterX = image.cols / 2;
        opticalCenterY = image.rows / 2;
#ifdef FACIAL_LANDMARK_ESTIMATION_DEBUG
        cerr << "Setting the optical center to (" << opticalCenterX << ", " << opticalCenterY << ")" << endl;
#endif
    }

    current_image = cv_image<bgr_pixel>(image);

    faces = detector(current_image);

    // Find the pose of each face.
    shapes.clear();
    for (auto face : faces){
        shapes.push_back(pose_model(current_image, face));
    }

#ifdef FACIAL_LANDMARK_ESTIMATION_DEBUG
    // Draws the contours of the face and face features onto the image

    _debug = image.clone();

    auto color = Scalar(0,128,128);

    for (unsigned long i = 0; i < shapes.size(); ++i)
    {
        const full_object_detection& d = shapes[i];

        for (unsigned long i = 1; i <= 16; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);

        for (unsigned long i = 28; i <= 30; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);

        for (unsigned long i = 18; i <= 21; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        for (unsigned long i = 23; i <= 26; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        for (unsigned long i = 31; i <= 35; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        line(_debug, toCv(d.part(30)), toCv(d.part(35)), color, 2, CV_AA);

        for (unsigned long i = 37; i <= 41; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        line(_debug, toCv(d.part(36)), toCv(d.part(41)), color, 2, CV_AA);

        for (unsigned long i = 43; i <= 47; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        line(_debug, toCv(d.part(42)), toCv(d.part(47)), color, 2, CV_AA);

        for (unsigned long i = 49; i <= 59; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        line(_debug, toCv(d.part(48)), toCv(d.part(59)), color, 2, CV_AA);

        for (unsigned long i = 61; i <= 67; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
        line(_debug, toCv(d.part(60)), toCv(d.part(67)), color, 2, CV_AA);
    }
#endif
}

landmarks_t FacialLandmarkEstimation::landmarks(size_t face_idx) const
{
    std::vector<Point2f> detected_points;

    for (unsigned long i = 0; i < 68; ++i) {
        detected_points.push_back(coordsOf(face_idx, i));
    }

    auto stomion = (coordsOf(face_idx, MOUTH_CENTER_TOP) + coordsOf(face_idx, MOUTH_CENTER_BOTTOM)) * 0.5;
    detected_points.push_back(stomion);

    return detected_points;
}

std::vector<landmarks_t> FacialLandmarkEstimation::all_landmarks() const
{
    std::vector<landmarks_t> all;

    for (auto i = 0; i < faces.size(); i++){
        all.push_back(landmarks(i));
    }

    return all;
}


Point2f FacialLandmarkEstimation::coordsOf(size_t face_idx, FACIAL_FEATURE feature) const
{
    return toCv(shapes[face_idx].part(feature));
}
Point2f FacialLandmarkEstimation::coordsOf(size_t face_idx, unsigned long index) const
{
    return toCv(shapes[face_idx].part(index));
}
