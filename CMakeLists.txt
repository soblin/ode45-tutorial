cmake_minimum_required(VERSION 3.12)

project(ode45_tutorial
  VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(matplotlibcpp17)
add_subdirectory(xtl)
add_subdirectory(xtensor)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC matplotlibcpp17::matplotlibcpp17 xtensor)
