cmake_minimum_required(VERSION 3.5.1)
project(SFM)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

#set(LIBELAS_SRC_DIR elas)
set(CMAKE_CXX_FLAGS "-msse3")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")


# include directory
#include_directories("${LIBELAS_SRC_DIR}")
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include ${PROJECT_SOURCE_DIR})
include_directories($(OpenCV_INCLUDE_DIRS))

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# sources
#FILE(GLOB LIBELAS_SRC_FILES "elas/*.cpp")
file(GLOB SOURCES "src/*.cpp")

add_library(sfm_support ${SOURCES})
target_link_libraries(sfm_support ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(sfm ${SOURCES})
target_link_libraries(sfm sfm_support ${OpenCV_LIBS} ${PCL_LIBRARIES})
