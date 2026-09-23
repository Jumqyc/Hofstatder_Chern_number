cmake_minimum_required(VERSION 3.24)

project(hofstadter LANGUAGES CXX)

# Default to an optimised build when the user did not pick one explicitly.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
option(HOF_NATIVE  "Compile with -march=native" ON)
option(HOF_PROFILE "Enable the HOF_PROFILE timing instrumentation" OFF)
set(HOF_EIGEN_TAG "3.4.1" CACHE STRING "Eigen git tag fetched from GitLab")

# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

include(FetchContent)
# Eigen is header-only for our purposes; turn its extra build products off.
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING         OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        ${HOF_EIGEN_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(Eigen3)

# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------
add_executable(hofstadter hofstadter.cpp)
set_target_properties(hofstadter PROPERTIES
    CXX_STANDARD          17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS        OFF)
target_include_directories(hofstadter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(hofstadter PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

if(HOF_NATIVE)
    target_compile_options(hofstadter PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>)
endif()
if(HOF_PROFILE)
    target_compile_definitions(hofstadter PRIVATE HOF_PROFILE)
endif()
