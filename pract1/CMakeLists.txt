cmake_minimum_required(VERSION 3.20)
project(pract1)

set(CMAKE_CXX_STANDARD 17)

# Find MPI
find_package(MPI REQUIRED)

# Add executables
add_executable(pract1  main.cpp)
target_link_libraries(pract1 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
    set_target_properties(pract1 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(pract1 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
