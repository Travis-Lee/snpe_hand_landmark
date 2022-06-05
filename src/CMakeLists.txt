cmake_minimum_required( VERSION 2.8 )
#setting opencv path 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hand_landmark)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../snpe/include/zdl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../config)

link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../snpe/lib/aarch64-android-clang6.0)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv-arm64-v8a/sdk/native/jni)

find_package(OpenCV REQUIRED core highgui imgproc)


add_library(GHL SHARED 
                hand_landmark/hand_landmark.cpp 
                g_hand_landmark_api.cpp)
target_compile_options(GHL PRIVATE -std=c++11 -O3 -frtti -fexceptions)

target_link_libraries(GHL PUBLIC config 
		                         libSNPE.so
                                 log
                                 z
                                 ${OpenCV_LIBS})












