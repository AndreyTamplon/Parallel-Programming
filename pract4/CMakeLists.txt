cmake_minimum_required(VERSION 3.21)
project(pract4)

set(CMAKE_CXX_STANDARD 14)

# Find MPI
find_package(MPI REQUIRED)

# Add executables
add_executable(pract4 point_point_communications.cpp)
target_link_libraries(pract4 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
    set_target_properties(pract4 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(pract4 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
