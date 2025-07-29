cmake_minimum_required(VERSION 3.22.1)

project(rerun)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(rerun REQUIRED)


include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

include_directories("/usr/include/eigen3")
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/Sophus)


add_executable(rerun_trajectoryError rerun_trajectoryError.cpp)
target_link_libraries(rerun_trajectoryError ${Sophus_LIBRARIES})
target_link_libraries(rerun_trajectoryError PRIVATE rerun_sdk)

add_executable(rerun_test rerun_test.cpp)
target_link_libraries(rerun_test PRIVATE rerun_sdk)

add_executable(rerun_Error rerun_Error.cpp)
target_link_libraries(rerun_trajectoryError ${Sophus_LIBRARIES})
target_link_libraries(rerun_Error PRIVATE rerun_sdk)