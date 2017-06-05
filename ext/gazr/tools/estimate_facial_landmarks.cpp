#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>

#include "LinearMath/Matrix3x3.h"

#include "../src/facial_landmark_estimation.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace po = boost::program_options;


int countCameras()
{
    VideoCapture temp_camera;
    int max_tested = 2;

    for (int i = 0; i < max_tested; ++i) {
        VideoCapture temp_camera(i);
        bool res = (!temp_camera.isOpened());
        temp_camera.release();
        if (res) return i;
    }

    return max_tested;
}


int main(int argc, char **argv)
{
    Mat frame;

    bool show_frame = false;
    bool use_camera = false;

    po::positional_options_description p;
    p.add("image", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("version,v", "shows version and exits")
            ("show,s", "display the image with gaze estimation")
            ("model", po::value<string>(), "dlib's trained face model")
            ("image", po::value<string>(), "image to process (png, jpg)")
            ("camera", po::value<int>()->default_value(0), "index of the webcam to use")
            ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                        .options(desc)
                        .positional(p)
                        .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << " " <<  STR(GAZR_VERSION) << "\n\n" << desc << "\n";
        return 1;
    }

    if (vm.count("version")) {
        cout << argv[0] << " " << STR(GAZR_VERSION) << "\n";
        return 0;
    }

    if (vm.count("show")) {
        show_frame = true;
    }

    if (vm.count("model") == 0) {
        cout << "You must specify the path to a trained dlib's face model\n"
             << "with the option --model." << endl;
        return 1;
    }

    if (vm.count("image") == 0) {
        use_camera = true;
    }

    auto estimator = FacialLandmarkEstimation(vm["model"].as<string>());

    VideoCapture video_in;

    if (use_camera) {
        video_in = VideoCapture(vm["camera"].as<int>());

        // adjust for your webcam!
        video_in.set(CV_CAP_PROP_FRAME_WIDTH, 320);
        video_in.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
        estimator.focalLength = 455;
        estimator.opticalCenterX = 160;
        estimator.opticalCenterY = 120;

        if(!video_in.isOpened()) {
            cerr << "Couldn't open camera" << endl;
            return 1;
        }
    }

    else {
        auto image = vm["image"].as<string>();

#ifdef OPENCV3
        frame = imread(image,IMREAD_COLOR);
#else
        frame = imread(image,CV_LOAD_IMAGE_COLOR);
#endif

        resize(frame, frame, Size(0,0), 0.2,0.2);

        estimator.focalLength = 85.0 / 22.3 * frame.size().width;
    }

    while(true) {
        if(use_camera) {
            auto ok = video_in.read(frame);
            if (!ok) break;
        }


        estimator.update(frame);


        auto all_landmarks = estimator.all_landmarks();

        int i = 0;
        cout << "{";

        for(auto landmarks : all_landmarks) {
            cout << "\"face_" << i << "\": [";
            cout << setprecision(4) << fixed;
            for (unsigned long j = 0; j < 69; ++j) {
                if (j > 0) cout << ", ";
                cout << "[" << landmarks[j].x << ", " << landmarks[j].y << "]";
            }
            cout << "],";

            i++;
        }
        cout << "}\n" << flush;

        if (show_frame) {
            Mat flipped = estimator._debug.clone();
            flip(estimator._debug, flipped, 1);
            imshow("landmarks", flipped);
            if(use_camera) {
                waitKey(10);
            }
            else {
                while(waitKey(10) != 1048603) {}
                break;
            }
        }
    }

}
