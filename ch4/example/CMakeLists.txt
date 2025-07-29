cmake_minimum_required(VERSION 2.8.12)

project(example)

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

# Eigen include
include_directories("/usr/include/eigen3")
# Sophus include
include_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/Sophus)
# Pangolin include
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(trajectoryError trajectoryError.cpp)
target_link_libraries(trajectoryError 
                        ${Sophus_INCLUDE_DIRS}
                        ${Pangolin_LIBRARIES}
                    )

# add_subdirectory(example)