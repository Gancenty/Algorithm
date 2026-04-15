cmake_minimum_required(VERSION 3.10)
project(algorithms)

find_package(Eigen3 REQUIRED)

add_executable(main main.cpp)
target_include_directories(main PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)