cmake_minimum_required(VERSION 3.12)
project(epipolar_visualizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.2 REQUIRED COMPONENTS
  core imgcodecs imgproc highgui features2d calib3d freetype)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(epipolar_viz
  src/app/main.cpp
  src/core/feature.cpp
  src/core/matching.cpp
  src/core/geometry.cpp
  src/core/draw.cpp
  src/ui/panel.cpp
  src/teach/teach.cpp
)

target_link_libraries(epipolar_viz ${OpenCV_LIBS})
