# CS231A-Screen-Stabilization

## Setup

### Compile Gazr
Gazr is one of the off-the-shelf head pose tracking libraries available.

First, navigate from the root of this repository to the `gazr` directory to make the `build` directory:

```
cd ext/gazr
mkdir build
cd build
```

Next, download and install dlib:

```
wget dlib.net/files/dlib-19.4.zip
unzip dlib-19.4.zip
rm dlib-19.4.zip
cd dlib-19.4
mkdir build
cd build
cmake ..
cmake --build . --config Release
make
sudo make install
```

Then go back to the build directory for Gazr:

```
cd ../..
```

Finally, compile Gazr:

```
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Now you can run the `demo_head_pose.py` script in the `src` directory.

### Install Vispy
Vispy is used for the renderer.

First, install pyglet:

```
sudo pip install pyglet
```

Next, install the latest development of vispy:

```
sudo pip install vispy
```

Now you can run the `demo_vispy.py` script in the `src` directory.

