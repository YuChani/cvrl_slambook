cmake_minimum_required(VERSION 3.22.1)

project(imagesBasics)
set(CMAKE_CXX_FALGS "-std=c++14")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imagesBasics imagesBasics.cpp)
target_include_directories(imagesBasics PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(imagesBasics ${OpenCV_LIBS})

add_executable(undistortImages undistortImages.cpp)
target_include_directories(undistortImages PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(undistortImages ${OpenCV_LIBS})
