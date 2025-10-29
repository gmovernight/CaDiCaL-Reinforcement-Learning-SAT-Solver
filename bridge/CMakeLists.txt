cmake_minimum_required(VERSION 3.16)
project(satrl_bridge LANGUAGES CXX)

# --- Guard against double inclusion (fixes duplicate target error) ---
if (TARGET satrl_bridge)
  message(STATUS "Target 'satrl_bridge' already exists; skipping duplicate definition.")
  return()
endif()

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python (use the venv Python if provided via -DPython3_EXECUTABLE)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

include(FetchContent)

# Try system/venv pybind11 first; if not found, fetch it.
find_package(pybind11 CONFIG QUIET)
if (NOT pybind11_FOUND)
  message(STATUS "pybind11 not found; fetching with FetchContent")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Build the module
pybind11_add_module(satrl_bridge pybridge.cpp)

# Link against in-tree CaDiCaL
target_include_directories(satrl_bridge PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../cadical/src
)

# If libcadical.a was not built with -fPIC, rebuild CaDiCaL with:
#   cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j
target_link_libraries(satrl_bridge PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../cadical/build/libcadical.a
)