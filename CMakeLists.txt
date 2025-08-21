cmake_minimum_required(VERSION 2.8.12)

project(TOP)

add_compile_options(-Wno-error=deprecated-copy)

# add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/g2o")
add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/Pangolin")
add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/Sophus")
# add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/ceres-solver")
# add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/DBoW3")

add_subdirectory("${CMAKE_SOURCE_DIR}/ch2")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch3")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch4")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch5")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch6")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch7")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch8")
add_subdirectory("${CMAKE_SOURCE_DIR}/rerun")
