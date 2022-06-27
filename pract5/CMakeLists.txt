cmake_minimum_required(VERSION 3.21)
project(pract5)

set(CMAKE_CXX_STANDARD 14)

# Find MPI
find_package(MPI REQUIRED)

# Add executables
add_executable(pract5 main.cpp)
target_link_libraries(pract5 ${MPI_LIBRARIES})

# Include MPI includes
include_directories(${MPI_INCLUDE_PATH})
if(MPI_COMPILE_FLAGS)
    set_target_properties(pract5 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(pract5 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
