cmake_minimum_required(VERSION 3.21)
project(pract2)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)
add_executable(pract2 "main.cpp")
target_link_libraries(pract2 PRIVATE OpenMP::OpenMP_CXX)