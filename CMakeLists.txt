cmake_minimum_required(VERSION 2.8.12)

project(TOP)

add_compile_options(-Wno-error=deprecated-copy)

add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/g2o")
add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/Pangolin")
add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/Sophus")

add_subdirectory("${CMAKE_SOURCE_DIR}/ch2")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch3")
add_subdirectory("${CMAKE_SOURCE_DIR}/ch4")
add_subdirectory("${CMAKE_SOURCE_DIR}/rerun")