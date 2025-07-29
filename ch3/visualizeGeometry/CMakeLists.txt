cmake_minimum_required(VERSION 2.8.12)

find_package(Pangolin REQUIRED)

project(visualizeGeometry)

include_directories("/usr/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(visualizeGeometry visualizeGeometry.cpp)
target_link_libraries(visualizeGeometry ${Pangolin_LIBRARIES})






