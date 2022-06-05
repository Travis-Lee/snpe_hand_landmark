cmake_minimum_required( VERSION 2.8 )
project(AI-Engine)
  
add_definitions("-Wall -g")
set(CMAKE_BUILD_TYPE Release)
   
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
   add_definitions(-Ofast)
   add_definitions(-ffast-math)
else()
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -ggdb")
endif()


add_definitions(-Wall -Wextra -Wno-unused-function)
add_definitions(-fPIC)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")

add_subdirectory(config)
add_subdirectory(src)
add_subdirectory(example)
