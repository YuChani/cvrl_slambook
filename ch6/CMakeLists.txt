cmake_minimum_required(VERSION 3.22.1)

project(chapter6)

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

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} Ceres::ceres)


add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} 
                Ceres::ceres
                fmt::fmt
                g2o::core
                g2o::stuff
                g2o::solver_dense
                g2o::types_slam3d)
