cmake_minimum_required(VERSION 3.16)
project(cadical LANGUAGES CXX)

# Use C++17 for consistency with the bridge.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build a static library with position independent code (pass -DCMAKE_POSITION_INDEPENDENT_CODE=ON).
option(CMAKE_POSITION_INDEPENDENT_CODE "Build with -fPIC" ON)

# Sources: all .cpp in src except the two application entry points.
file(GLOB CADICAL_ALL_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(CADICAL_EXCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cadical.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mobical.cpp"
)

set(CADICAL_SOURCES "${CADICAL_ALL_CPP}")
list(REMOVE_ITEM CADICAL_SOURCES ${CADICAL_EXCLUDE})

add_library(cadical STATIC ${CADICAL_SOURCES})
target_include_directories(cadical PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Avoid dependency on generated build.hpp (manual build mode per BUILD.md)
target_compile_definitions(cadical PRIVATE NBUILD)

# Reasonable defaults for Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optional: build the standalone solver binary for quick sanity tests.
add_executable(cadical_app ${CMAKE_CURRENT_SOURCE_DIR}/src/cadical.cpp)
target_link_libraries(cadical_app PRIVATE cadical)
target_include_directories(cadical_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
