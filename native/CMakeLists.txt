cmake_minimum_required(VERSION 3.10)
project(zed_oc_shim CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Installed by `sudo make install` in zed-open-capture: the header lands in
# <prefix>/include/zed-open-capture and the library in <prefix>/lib.
find_path(ZED_OC_INCLUDE_DIR videocapture.hpp PATH_SUFFIXES zed-open-capture)
find_library(ZED_OC_LIBRARY NAMES zed_open_capture)

if(NOT ZED_OC_INCLUDE_DIR OR NOT ZED_OC_LIBRARY)
  message(FATAL_ERROR
    "zed-open-capture was not found. Build and install it first:\n"
    "  git clone https://github.com/stereolabs/zed-open-capture\n"
    "  cd zed-open-capture && mkdir build && cd build\n"
    "  cmake .. -DBUILD_SENSORS=OFF -DBUILD_EXAMPLES=OFF\n"
    "  make -j$(nproc) && sudo make install && sudo ldconfig")
endif()

add_library(zed_oc_shim SHARED zed_oc_shim.cpp)
target_include_directories(zed_oc_shim PRIVATE ${ZED_OC_INCLUDE_DIR})
target_link_libraries(zed_oc_shim PRIVATE ${ZED_OC_LIBRARY})
