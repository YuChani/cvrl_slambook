cmake_minimum_required(VERSION 2.8.12)

project(chapter4)

find_package(Sophus REQUIRED)

# Eigen include
include_directories("/usr/include/eigen3")
# Sophus include
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/Sophus)

add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus ${Sophus_INCLUDE_DIRS})

add_subdirectory(example)