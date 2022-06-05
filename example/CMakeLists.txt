set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/snpe_src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../snpe/include/zdl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../snpe/lib/aarch64-android-clang6.0)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/opencv-arm64-v8a/sdk/native/jni)

find_package(OpenCV REQUIRED core highgui imgproc)

add_executable(hand_landmark test_hand_landmark.cpp)
target_link_libraries(hand_landmark GHL ${OpenCV_LIBS} libSNPE.so)

install(TARGETS hand_landmark DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/exe)
