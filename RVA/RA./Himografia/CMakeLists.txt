cmake_minimum_required (VERSION 3.5)
project (ARBlockT1)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 4.5 REQUIRED)

LINK_LIBRARIES(${OpenCV_LIBS})


add_executable(homog homog.cpp rva.cpp)





