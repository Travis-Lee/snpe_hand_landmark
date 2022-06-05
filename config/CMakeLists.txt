cmake_minimum_required( VERSION 2.8 )

add_definitions(-fPIC )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(config STATIC g_hand_config.cpp cJSON.cpp)

