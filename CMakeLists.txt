cmake_minimum_required(VERSION 3.12)
project(EpipolarViz LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(epipolar_viz
    src/main.cpp
  third_party/tinyfiledialogs/tinyfiledialogs.c
)

target_include_directories(epipolar_viz PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvui
    ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyfiledialogs
)

target_link_libraries(epipolar_viz PRIVATE
    ${OpenCV_LIBS}
)
