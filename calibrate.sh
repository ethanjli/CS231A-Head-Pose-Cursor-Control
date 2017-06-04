cd ext/StereoVision
python setup.py build
cp build/scripts-2.7/* build/lib.linux-x86_64-2.7/
cd build/lib.linux-x86_64-2.7
python capture_chessboards --rows 6 --columns 8 --square-size 2.54 --calibration-folder ../../../../calib 0 1 20 ../../../../calib/images
cd ../../../
