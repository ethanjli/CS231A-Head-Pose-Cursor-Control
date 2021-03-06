cmake_minimum_required(VERSION 2.8.3)
project(gazr)

set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

add_definitions(-std=c++11 -DGAZR_VERSION=${VERSION})

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

option(DEBUG "Enable debug visualizations" ON)
option(WITH_TOOLS "Compile sample tools" ON)
option(WITH_ROS "Build ROS nodes" OFF)

if(WITH_ROS)

    find_package(catkin REQUIRED COMPONENTS
        roscpp
        tf
        std_msgs
        visualization_msgs
        sensor_msgs
        cv_bridge
        image_transport
        image_geometry
        )

    include_directories(${catkin_INCLUDE_DIRS})

    catkin_package(
    CATKIN_DEPENDS
        tf
    DEPENDS OpenCV
    LIBRARIES
    )
endif()

if(DEBUG)
    find_package(OpenCV COMPONENTS core imgproc calib3d highgui REQUIRED)
else()
    find_package(OpenCV COMPONENTS core imgproc calib3d REQUIRED)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(${OpenCV_VERSION} VERSION_GREATER 2.9.0)
    set(OPENCV3 TRUE)
    add_definitions(-DOPENCV3)
endif()



if(DEBUG)
    add_definitions(-DHEAD_POSE_ESTIMATION_DEBUG -DFACIAL_LANDMARK_ESTIMATION_DEBUG)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(gazr SHARED src/head_pose_estimation.cpp src/facial_landmark_estimation.cpp)
target_link_libraries(gazr ${dlib_LIBRARIES} ${OpenCV_LIBRARIES})

if(WITH_ROS)

    add_executable(estimate_focus src/estimate_focus.cpp)
    target_link_libraries(estimate_focus ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

    add_executable(estimate src/head_pose_estimation_ros.cpp src/ros_head_pose_estimator.cpp)
    target_link_libraries(estimate gazr ${catkin_LIBRARIES})

    install(TARGETS estimate_focus gazr estimate
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

    install(FILES
        launch/gazr.launch
        launch/gazr_gscam.launch
        calib/logitech-c920_640x360.ini
        share/shape_predictor_68_face_landmarks.dat
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
    )
endif()

if(WITH_TOOLS)

    if(OPENCV3)
        find_package(OpenCV COMPONENTS core imgproc calib3d highgui imgcodecs videoio REQUIRED)
    else()
        find_package(OpenCV COMPONENTS core imgproc calib3d highgui REQUIRED)
    endif()

    find_package(Boost COMPONENTS program_options REQUIRED)

    add_executable(gazr_benchmark_head_pose_single_frame tools/benchmark_head_pose_estimation_single_frame.cpp)
    target_link_libraries(gazr_benchmark_head_pose_single_frame gazr ${OpenCV_LIBRARIES})

    add_executable(gazr_estimate_head_direction tools/estimate_head_direction.cpp)
    target_link_libraries(gazr_estimate_head_direction gazr ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

    add_executable(gazr_estimate_facial_landmarks tools/estimate_facial_landmarks.cpp)
    target_link_libraries(gazr_estimate_facial_landmarks gazr ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

    add_executable(gazr_show_head_pose tools/show_head_pose.cpp)
    target_link_libraries(gazr_show_head_pose gazr ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

endif()


