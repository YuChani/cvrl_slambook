cmake_minimum_required(VERSION 3.22.1)

project(stero)
set(CMAKE_CXX_FALGS "-std=c++14")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Eigen include
include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(steroVision steroVision.cpp)
target_include_directories(steroVision PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(steroVision 
                        ${OpenCV_LIBS}
                        ${Pangolin_LIBRARIES}
                        )
