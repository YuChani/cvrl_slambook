cmake_minimum_required(VERSION 3.22.1)

project(rgbd)

find_package(Sophus REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/Sophus)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(jointMap jointMap.cpp)
target_include_directories(jointMap PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jointMap 
                        ${OpenCV_LIBS}
                        ${Sophus_INCLUDE_DIRS}
                        ${Pangolin_LIBRARIES}
                    )