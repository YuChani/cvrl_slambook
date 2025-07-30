cmake_minimum_required(VERSION 3.22.1)

project(chapter5)

set(CMAKE_CXX_FALGS "-std=c++14")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")


add_subdirectory(imagesBasics)
add_subdirectory(rgbd)
add_subdirectory(stero)