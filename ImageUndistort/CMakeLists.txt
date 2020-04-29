cmake_minimum_required(VERSION 3.14)
project(ImageUndistort)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "/home/jiang/6_lib/install/opencv3.3.0/share/OpenCV")

find_package(OpenCV REQUIRED)

add_executable(undistort2_1 undistort2_1.cpp)
target_link_libraries(undistort2_1
        ${OpenCV_LIBS}
)

add_executable(undistort2_2 undistort2_2.cpp)
target_link_libraries(undistort2_2
        ${OpenCV_LIBS}
        )

add_executable(undistort2_3_1 undistort2_3_1.cpp)
target_link_libraries(undistort2_3_1
        ${OpenCV_LIBS}
        )

add_executable(undistort2_3_2 undistort2_3_2.cpp)
target_link_libraries(undistort2_3_2
        ${OpenCV_LIBS}
        )

add_executable(undistort2_4 undistort2_4.cpp)
target_link_libraries(undistort2_4
        ${OpenCV_LIBS}
        )