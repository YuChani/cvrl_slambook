cmake_minimum_required(VERSION 3.22.1)

project(chapter8)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(fmt REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

# g2o
find_package(g2o REQUIRED)
include_directories(${g2o_INCLUDE_DIRS})

# Sophus include
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/Sophus)

# Pangolin include
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(opencv_optical_flow opencv_optical_flow.cpp)
target_link_libraries(opencv_optical_flow ${OpenCV_LIBS})

add_executable(compare_optical_vs_matching compare_optical_vs_matching.cpp)
target_link_libraries(compare_optical_vs_matching ${OpenCV_LIBS})

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} fmt::fmt ${Pangolin_LIBRARIES} ${Sophus_INCLUDE_DIRS})